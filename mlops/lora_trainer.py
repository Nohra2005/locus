"""
lora_trainer.py — LoRA fine-tuning of fashion-CLIP vision encoder.

Architecture:
  We inject LoRA adapters (rank r=4) into the q_proj and v_proj of every
  attention layer in the CLIP vision transformer. The base model weights stay
  frozen. Only ~150K parameters are trained vs 86M total.

Loss:
  InfoNCE (NT-Xent / SimCLR loss). For a batch of N (anchor, positive) pairs:
    - anchor[i]   = augmented product image  (simulates consumer/internet photo)
    - positive[i] = original catalog image   (clean studio shot)
  The loss pulls (anchor[i], positive[i]) together while treating all other
  items in the batch as negatives. Temperature τ=0.07 is standard for CLIP.

CPU config:
  r=4, all attention blocks, batch_size=8, gradient accumulation=4
  (effective batch = 32). Adam lr=1e-4. ~1-2h per 300 steps on CPU.

Usage:
  from lora_trainer import train
  adapter_path = train(manifest_path="pairs_cache/pairs_manifest.json",
                       output_dir="lora_runs/run_001",
                       mlflow_run_id="abc123")
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import mlflow
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME        = "patrickjohncyh/fashion-clip"
LORA_R            = 4
LORA_ALPHA        = 8       # scaling = alpha / r = 2.0
LORA_DROPOUT      = 0.05
TEMPERATURE       = 0.07
LEARNING_RATE     = 1e-4
BATCH_SIZE        = int(os.getenv("BATCH_SIZE", 4))
GRAD_ACCUM_STEPS  = int(os.getenv("GRAD_ACCUM_STEPS", 4))  # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
MAX_STEPS         = int(os.getenv("MAX_TRAIN_STEPS", 300))
LOG_EVERY         = int(os.getenv("LOG_EVERY_STEPS", 20))  # log loss to MLflow every N steps
# ─────────────────────────────────────────────────────────────────────────────


def infonce_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    NT-Xent loss for a batch of positive pairs (z_i[k], z_j[k]).

    Both tensors are already L2-normalised embeddings of shape (B, D).
    In-batch negatives: for anchor z_i[k], the positive is z_j[k] and all
    other z_j[m] (m≠k) are negatives. We also use the symmetric direction
    (z_j → z_i) for a more stable gradient signal.
    """
    B = z_i.shape[0]

    # Concatenate both views → (2B, D)
    z = torch.cat([z_i, z_j], dim=0)

    # Full similarity matrix (2B × 2B), scaled by temperature
    sim = torch.mm(z, z.T) / temperature

    # Mask self-similarity on the diagonal
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim  = sim.masked_fill(mask, float("-inf"))

    # Positive indices: for row k in [0,B) the positive is k+B; for k in [B,2B) it's k-B
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(B,        device=z.device),
    ])

    return F.cross_entropy(sim, labels)


def _embed_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    pil_images: list,
) -> torch.Tensor:
    """
    Run a list of PIL images through vision_model + visual_projection.
    Returns L2-normalised embeddings of shape (N, 512).

    Note: torch.no_grad() is intentionally NOT used here during training
    so gradients flow through the LoRA adapters.
    """
    inputs = processor(images=pil_images, return_tensors="pt", padding=True)
    vision_out     = model.vision_model(**inputs)
    image_features = model.visual_projection(vision_out.pooler_output)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features


def _load_pairs(manifest_path: str) -> list[dict]:
    with open(manifest_path) as f:
        pairs = json.load(f)
    # Verify files exist, skip missing
    valid = [p for p in pairs
             if Path(p["anchor_path"]).exists() and Path(p["positive_path"]).exists()]
    if len(valid) < len(pairs):
        print(f"[TRAINER] Skipped {len(pairs) - len(valid)} pairs with missing files")
    return valid


def train(
    manifest_path:  str,
    output_dir:     str,
    mlflow_run_id:  Optional[str] = None,
) -> str:
    """
    Fine-tune fashion-CLIP with LoRA using the pairs in manifest_path.

    Args:
        manifest_path: Path to pairs_manifest.json from build_training_pairs.py
        output_dir:    Directory to save the LoRA adapter weights
        mlflow_run_id: If provided, log metrics into this existing MLflow run

    Returns:
        Path to the saved LoRA adapter directory.
    """
    try:
        from peft import get_peft_model, LoraConfig
    except ImportError:
        raise ImportError(
            "peft is required for LoRA training. "
            "Install with: pip install peft"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_pairs(manifest_path)
    if not pairs:
        raise ValueError(f"No valid training pairs found in {manifest_path}")
    print(f"[TRAINER] Loaded {len(pairs)} training pairs")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[TRAINER] Loading {MODEL_NAME}...")
    model     = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.train()

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA only to the vision encoder (q_proj, v_proj in all attention blocks)
    # We target the vision model only — text encoder stays untouched since we're
    # optimising for image-to-image retrieval, not image-text alignment.
    lora_config = LoraConfig(
        r            = LORA_R,
        lora_alpha   = LORA_ALPHA,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout = LORA_DROPOUT,
        bias         = "none",
        inference_mode = False,
    )
    model.vision_model = get_peft_model(model.vision_model, lora_config)

    # Also unfreeze visual_projection (512→512 linear) — cheap but meaningful
    for param in model.visual_projection.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[TRAINER] Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    step          = 0
    accum_loss    = 0.0
    accum_batches = 0
    start_time    = time.time()

    # Pair index cycles if we have fewer pairs than MAX_STEPS * BATCH_SIZE
    pair_idx = 0

    print(f"[TRAINER] Starting training: max_steps={MAX_STEPS}, "
          f"batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS}")

    optimizer.zero_grad()

    while step < MAX_STEPS:
        # Build batch
        anchors   = []
        positives = []
        for _ in range(BATCH_SIZE):
            p = pairs[pair_idx % len(pairs)]
            pair_idx += 1
            try:
                anchors.append(Image.open(p["anchor_path"]).convert("RGB"))
                positives.append(Image.open(p["positive_path"]).convert("RGB"))
            except Exception as e:
                print(f"[TRAINER] Skipping bad pair: {e}")
                continue

        if len(anchors) < 2:
            # Need at least 2 pairs for meaningful in-batch negatives
            continue

        try:
            z_i = _embed_batch(model, processor, anchors)
            z_j = _embed_batch(model, processor, positives)
            loss = infonce_loss(z_i, z_j, TEMPERATURE) / GRAD_ACCUM_STEPS
            loss.backward()
        except Exception as e:
            print(f"[TRAINER] Step {step} failed: {e}")
            optimizer.zero_grad()
            continue

        accum_loss    += loss.item() * GRAD_ACCUM_STEPS  # un-scale for logging
        accum_batches += 1

        # Gradient accumulation: only step optimizer every GRAD_ACCUM_STEPS batches
        if accum_batches % GRAD_ACCUM_STEPS == 0:
            # Clip gradients to prevent exploding on CPU with small batches
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            avg_loss = accum_loss / GRAD_ACCUM_STEPS
            accum_loss = 0.0

            if step % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                eta     = (elapsed / step) * (MAX_STEPS - step)
                print(f"[TRAINER] step={step}/{MAX_STEPS}  loss={avg_loss:.4f}  "
                      f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")
                if mlflow_run_id:
                    mlflow.log_metric("train_loss", avg_loss, step=step)

    # ── Save LoRA adapter ─────────────────────────────────────────────────────
    adapter_dir = output_dir / "lora_adapter"
    model.vision_model.save_pretrained(str(adapter_dir))

    # Also save visual_projection weights separately (not part of PEFT adapter)
    torch.save(
        model.visual_projection.state_dict(),
        output_dir / "visual_projection.pt",
    )

    elapsed = time.time() - start_time
    print(f"\n[TRAINER] Training complete in {elapsed/60:.1f} min")
    print(f"[TRAINER] LoRA adapter saved to: {adapter_dir}")

    return str(adapter_dir)
