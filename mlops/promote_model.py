"""
promote_model.py — Evaluate a new LoRA adapter and promote/rollback.

Evaluation runs entirely offline (no running service needed):
  1. Load CLIPModel + new LoRA adapter in this process
  2. Re-embed golden dataset query images with the new model
  3. Search Qdrant directly with the new embeddings
  4. Compute hit@5 and ACS@5 against golden dataset ground truth
  5. Compare to MLflow baseline

Promotion:
  - Copies adapter to visual_engine/models/lora_adapter/
  - Copies visual_projection.pt to visual_engine/models/
  - Calls POST /reload-adapter on visual engine (hot reload, no restart)
  - Calls POST /reindex on gateway to re-embed catalog with new model
  - Registers adapter in MLflow Model Registry
"""

import base64
import io
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Optional

import mlflow
import requests
import torch

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL           = os.getenv("QDRANT_URL")
QDRANT_API_KEY       = os.getenv("QDRANT_API_KEY")
QDRANT_HOST          = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT          = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME      = "locus_items"
MLFLOW_URI           = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
GATEWAY_URL          = os.getenv("GATEWAY_URL",   "http://localhost:8000")
VISUAL_ENGINE_URL    = os.getenv("VISUAL_HOST",   "http://localhost:8001")

SCRIPT_DIR           = Path(__file__).parent
GOLDEN_DATASET_PATH  = SCRIPT_DIR / "golden_dataset.json"
VISUAL_ENGINE_MODELS = Path(os.getenv(
    "VISUAL_ENGINE_MODELS",
    SCRIPT_DIR.parent / "visual_engine" / "models"
))
# Path to the models dir as seen by the visual_engine container (different mount point)
VISUAL_ENGINE_MODELS_INTERNAL = os.getenv(
    "VISUAL_ENGINE_MODELS_INTERNAL",
    str(VISUAL_ENGINE_MODELS),  # fallback: same path (e.g. running locally)
)
MODEL_NAME           = "patrickjohncyh/fashion-clip"
TOP_K                = 5
# ─────────────────────────────────────────────────────────────────────────────


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _fetch_image_bytes(url: str) -> bytes:
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
        return base64.b64decode(b64)
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Locus-Evaluator/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def _embed_with_adapter(
    model,
    processor,
    pil_image,
) -> list[float]:
    """Embed a single PIL image using the given model. Returns a list of floats."""
    from PIL import Image
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        vision_out     = model.vision_model(**inputs)
        image_features = model.visual_projection(vision_out.pooler_output)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features[0].tolist()


def evaluate_adapter(
    adapter_dir:            str,
    visual_projection_path: Optional[str] = None,
) -> tuple[float, float]:
    """
    Evaluate a LoRA adapter on the golden dataset.

    Loads the adapter into a fresh CLIPModel (does not touch the running
    visual_engine), re-embeds all golden dataset queries, searches Qdrant,
    and returns (hit@5, acs@5).
    """
    from peft import PeftModel
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image

    print(f"[PROMOTE] Loading model with adapter from: {adapter_dir}")
    model     = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Apply LoRA adapter
    model.vision_model = PeftModel.from_pretrained(model.vision_model, adapter_dir)

    # Load visual_projection if separately saved
    if visual_projection_path and Path(visual_projection_path).exists():
        proj_state = torch.load(visual_projection_path, map_location="cpu")
        model.visual_projection.load_state_dict(proj_state)

    model.eval()

    # Load golden dataset
    with open(GOLDEN_DATASET_PATH) as f:
        golden = json.load(f)
    print(f"[PROMOTE] Evaluating on {len(golden)} golden queries...")

    qdrant_client = _get_qdrant_client()

    hits      = []
    acs_scores = []

    for entry in golden:
        query_url     = entry.get("query_image_url", "")
        relevant_ids  = set(entry.get("relevant_product_ids", []))
        category_tag  = entry.get("query_category_tag", "")

        if not query_url or not relevant_ids:
            continue

        # Rewrite localhost URLs to the Docker service hostname so the mlops
        # container can reach the gateway (localhost doesn't resolve inside Docker).
        if query_url.startswith("http://localhost:"):
            query_url = query_url.replace("http://localhost:8000", GATEWAY_URL.rstrip("/"), 1)

        # Embed query with new adapter
        try:
            img_bytes = _fetch_image_bytes(query_url)
            pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            vector    = _embed_with_adapter(model, processor, pil_img)
        except Exception as e:
            print(f"  [SKIP] Could not embed query {query_url[:50]}: {e}")
            continue

        # Search Qdrant
        from qdrant_client.http import models as qmodels
        query_filter = None
        if category_tag:
            query_filter = qmodels.Filter(must=[
                qmodels.FieldCondition(
                    key="category_tag",
                    match=qmodels.MatchValue(value=category_tag)
                )
            ])

        try:
            search_result = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                query_filter=query_filter,
                limit=TOP_K,
                with_vectors=False,
            )
            results = search_result.points
        except Exception as e:
            print(f"  [SKIP] Qdrant search failed: {e}")
            continue

        # hit@5
        result_ids = [r.payload.get("product_id", str(r.id)) for r in results]
        hit = int(any(pid in relevant_ids for pid in result_ids))
        hits.append(hit)

        # ACS@5 — average cosine similarity of top-5 (Qdrant scores are already cosine)
        if results:
            avg_score = sum(r.score for r in results) / len(results)
            acs_scores.append(avg_score)

        time.sleep(0.05)  # avoid overwhelming Qdrant

    hit5 = sum(hits) / len(hits) if hits else 0.0
    acs5 = sum(acs_scores) / len(acs_scores) if acs_scores else 0.0

    print(f"[PROMOTE] Evaluation results: hit@5={hit5:.4f}, ACS@5={acs5:.4f} "
          f"(n={len(hits)} queries)")
    return hit5, acs5


def promote_adapter(
    adapter_dir:            str,
    visual_projection_path: Optional[str],
    mlflow_run_id:          str,
) -> None:
    """
    Copy adapter to visual_engine/models/, trigger hot reload and re-indexing.
    """
    VISUAL_ENGINE_MODELS.mkdir(parents=True, exist_ok=True)
    dest_adapter = VISUAL_ENGINE_MODELS / "lora_adapter"
    dest_proj    = VISUAL_ENGINE_MODELS / "visual_projection.pt"

    # Back up existing adapter if present
    if dest_adapter.exists():
        backup = VISUAL_ENGINE_MODELS / "lora_adapter_prev"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(dest_adapter, backup)
        print(f"[PROMOTE] Backed up previous adapter to {backup}")

    # Copy new adapter
    if dest_adapter.exists():
        shutil.rmtree(dest_adapter)
    shutil.copytree(adapter_dir, dest_adapter)
    print(f"[PROMOTE] Copied adapter to {dest_adapter}")

    # Copy visual_projection weights
    if visual_projection_path and Path(visual_projection_path).exists():
        shutil.copy2(visual_projection_path, dest_proj)
        print(f"[PROMOTE] Copied visual_projection to {dest_proj}")

    # ── Hot reload visual engine ───────────────────────────────────────────
    print("[PROMOTE] Requesting visual engine adapter reload...")
    # Translate paths from the mlops container's perspective to the visual_engine's perspective
    internal_adapter = Path(VISUAL_ENGINE_MODELS_INTERNAL) / "lora_adapter"
    internal_proj    = Path(VISUAL_ENGINE_MODELS_INTERNAL) / "visual_projection.pt"
    try:
        resp = requests.post(
            f"{VISUAL_ENGINE_URL}/reload-adapter",
            json={
                "adapter_path":           str(internal_adapter),
                "visual_projection_path": str(internal_proj),
            },
            timeout=60,
        )
        resp.raise_for_status()
        print(f"[PROMOTE] Visual engine reloaded: {resp.json()}")
    except Exception as e:
        print(f"[PROMOTE] WARNING: Could not hot-reload visual engine: {e}")
        print("          Restart visual_engine container to apply new adapter.")

    # ── Trigger catalog re-indexing ───────────────────────────────────────
    # All existing product vectors are now stale (different embedding space).
    # The gateway's /reindex endpoint re-embeds all locus_items.
    print("[PROMOTE] Triggering catalog re-indexing (this may take several minutes)...")
    try:
        resp = requests.post(
            f"{GATEWAY_URL}/reindex",
            timeout=10,  # fire-and-forget: re-indexing runs async in gateway
        )
        resp.raise_for_status()
        print(f"[PROMOTE] Re-indexing started: {resp.json()}")
    except Exception as e:
        print(f"[PROMOTE] WARNING: Could not trigger re-indexing: {e}")
        print("          Run POST /reindex manually to update catalog vectors.")

    # ── Register in MLflow Model Registry ────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        client = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{mlflow_run_id}/lora_adapter"
        # Log adapter as MLflow artifact
        mlflow.log_artifacts(str(dest_adapter), artifact_path="lora_adapter")
        # Register
        mv = mlflow.register_model(model_uri=model_uri, name="locus_lora_adapter")
        client.set_registered_model_alias(
            name    = "locus_lora_adapter",
            alias   = "production",
            version = mv.version,
        )
        print(f"[PROMOTE] Registered as MLflow model version {mv.version} (alias: production)")
    except Exception as e:
        print(f"[PROMOTE] MLflow registration skipped: {e}")
