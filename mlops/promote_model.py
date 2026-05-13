"""
promote_model.py — Deploy a LoRA adapter to the visual engine.

Promotion:
  - Copies adapter to visual_engine/models/lora_adapter/
  - Copies visual_projection.pt to visual_engine/models/
  - Calls POST /reload-adapter on visual engine (hot reload, no restart)
  - Calls POST /reindex on gateway to re-embed catalog with new model
  - Registers adapter in MLflow Model Registry

Promotion decisions are made by retrain_clip.py based on val_recall@5 delta
from training (not a separate judge eval). This file only handles deployment.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

import mlflow
import requests

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI                    = os.getenv("MLFLOW_TRACKING_URI", "http://20.240.203.22:5000")
GATEWAY_URL                   = os.getenv("GATEWAY_URL",  "http://localhost:8000")
VISUAL_ENGINE_URL             = os.getenv("VISUAL_HOST",  "http://localhost:8001")
VISUAL_ENGINE_MODELS          = Path(os.getenv(
    "VISUAL_ENGINE_MODELS",
    str(Path(__file__).parent.parent / "visual_engine" / "models"),
))
VISUAL_ENGINE_MODELS_INTERNAL = os.getenv(
    "VISUAL_ENGINE_MODELS_INTERNAL",
    str(VISUAL_ENGINE_MODELS),
)
# ─────────────────────────────────────────────────────────────────────────────


def promote_adapter(
    adapter_dir:            str,
    visual_projection_path: Optional[str],
    mlflow_run_id:          str,
) -> None:
    """Copy adapter to visual_engine/models/, trigger hot reload and re-indexing."""
    VISUAL_ENGINE_MODELS.mkdir(parents=True, exist_ok=True)
    dest_adapter = VISUAL_ENGINE_MODELS / "lora_adapter"
    dest_proj    = VISUAL_ENGINE_MODELS / "visual_projection.pt"

    if dest_adapter.exists():
        backup = VISUAL_ENGINE_MODELS / "lora_adapter_prev"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(dest_adapter, backup)
        print(f"[PROMOTE] Backed up previous adapter to {backup}")

    if dest_adapter.exists():
        shutil.rmtree(dest_adapter)
    shutil.copytree(adapter_dir, dest_adapter)
    print(f"[PROMOTE] Copied adapter to {dest_adapter}")

    if visual_projection_path and Path(visual_projection_path).exists():
        shutil.copy2(visual_projection_path, dest_proj)
        print(f"[PROMOTE] Copied visual_projection to {dest_proj}")

    print("[PROMOTE] Requesting visual engine adapter reload...")
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

    print("[PROMOTE] Triggering catalog re-indexing...")
    try:
        resp = requests.post(f"{GATEWAY_URL}/reindex", timeout=10)
        resp.raise_for_status()
        print(f"[PROMOTE] Re-indexing started: {resp.json()}")
    except Exception as e:
        print(f"[PROMOTE] WARNING: Could not trigger re-indexing: {e}")
        print("          Run POST /reindex manually to update catalog vectors.")

    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        client    = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{mlflow_run_id}/lora_adapter"
        mlflow.log_artifacts(str(dest_adapter), artifact_path="lora_adapter")
        mv = mlflow.register_model(model_uri=model_uri, name="locus_lora_adapter")
        client.set_registered_model_alias(
            name    = "locus_lora_adapter",
            alias   = "production",
            version = mv.version,
        )
        print(f"[PROMOTE] Registered as MLflow model version {mv.version} (alias: production)")
    except Exception as e:
        print(f"[PROMOTE] MLflow registration skipped: {e}")
