"""
retrain_clip.py — Orchestrator for the automated LoRA retraining pipeline.

Pipeline:
  1. Check trigger conditions (feedback threshold OR quality drift)
  2. Build augmented training pairs from catalog images
  3. Train LoRA adapters on fashion-CLIP vision encoder
  4. Evaluate new adapter on golden dataset (offline, no service disruption)
  5. Promote if hit@5 >= baseline - 0.02, else rollback and log rejection

This script is designed to run:
  - On a schedule (every 48h) via the mlops_retrain Docker service
  - Manually via: python retrain_clip.py --force
  - Via gateway POST /trigger-retrain endpoint

All results are logged to MLflow under experiment "locus_lora_retraining".

Usage:
  python retrain_clip.py             # respects trigger thresholds
  python retrain_clip.py --force     # skip threshold checks, always train
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL           = os.getenv("QDRANT_URL")
QDRANT_API_KEY       = os.getenv("QDRANT_API_KEY")
QDRANT_HOST          = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT          = int(os.getenv("QDRANT_PORT", 6333))
FEEDBACK_COL         = "locus_feedback"
MLFLOW_URI           = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
GATEWAY_URL          = os.getenv("GATEWAY_URL", "http://localhost:8000")
OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY", "")
EXPERIMENT_NAME      = "locus_lora_retraining"

# Trigger thresholds
MIN_FEEDBACK_PAIRS   = int(os.getenv("MIN_FEEDBACK_PAIRS", 50))   # lower for dev/demo
MIN_ACS_THRESHOLD    = float(os.getenv("MIN_ACS_THRESHOLD", 0.70))

# Paths
SCRIPT_DIR           = Path(__file__).parent
RUNS_DIR             = SCRIPT_DIR / "lora_runs"
PAIRS_CACHE_DIR      = SCRIPT_DIR / "pairs_cache"
VISUAL_ENGINE_MODELS = Path(os.getenv(
    "VISUAL_ENGINE_MODELS",
    SCRIPT_DIR.parent / "visual_engine" / "models"
))
# ─────────────────────────────────────────────────────────────────────────────


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _count_feedback_pairs() -> int:
    """Count the number of actionable feedback pairs (positive + negative) in Qdrant."""
    try:
        client = _get_qdrant_client()
        from qdrant_client.http import models as qmodels
        pos_count = client.count(
            collection_name=FEEDBACK_COL,
            count_filter=qmodels.Filter(must=[
                qmodels.FieldCondition(
                    key="training_signal",
                    match=qmodels.MatchAny(any=["positive", "negative"])
                )
            ]),
            exact=True,
        ).count
        return pos_count
    except Exception as e:
        print(f"[RETRAIN] Could not count feedback: {e}")
        return 0


def _cleanup_stale_runs() -> None:
    """
    Mark any RUNNING runs in locus_lora_retraining as KILLED on startup.
    These are zombies left by previous container crashes or Docker restarts —
    MLflow never received an end signal when the process died.
    """
    import mlflow, time as _time
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            return
        stale = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
            max_results=100,
        )
        if stale:
            now_ms = int(_time.time() * 1000)
            for r in stale:
                client.set_terminated(r.info.run_id, status="KILLED", end_time=now_ms)
            print(f"[RETRAIN] Cleaned up {len(stale)} stale RUNNING run(s) from previous crash/restart")
    except Exception as e:
        print(f"[RETRAIN] Could not clean up stale runs: {e}")



def _check_triggers(force: bool) -> tuple[bool, str]:
    """
    Returns (should_train, reason).
    """
    if force:
        return True, "forced via --force flag"

    n_pairs = _count_feedback_pairs()
    print(f"[RETRAIN] Feedback pairs in Qdrant: {n_pairs} (threshold: {MIN_FEEDBACK_PAIRS})")

    if n_pairs >= MIN_FEEDBACK_PAIRS:
        return True, f"feedback threshold met ({n_pairs} >= {MIN_FEEDBACK_PAIRS})"

    # Check ACS@5 drift via latest vss_cache.json written by metrics_exporter
    vss_cache = SCRIPT_DIR / "vss_cache.json"
    if vss_cache.exists():
        try:
            with open(vss_cache) as f:
                data = json.load(f)
            acs = data.get("acs_at_5", 1.0)
            if acs < MIN_ACS_THRESHOLD:
                return True, f"ACS@5 quality drift detected ({acs:.3f} < {MIN_ACS_THRESHOLD})"
        except Exception:
            pass

    return False, f"no trigger condition met (pairs={n_pairs}, threshold={MIN_FEEDBACK_PAIRS})"


def run_pipeline(force: bool = False) -> dict:
    """
    Full retraining pipeline. Returns a results dict with outcome and metrics.
    """
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Create experiment if it doesn't exist
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Kill any zombie runs left by previous crashes before starting a new one
    _cleanup_stale_runs()

    run_name = f"lora_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"  Locus LoRA Retraining Pipeline")
    print(f"  Run: {run_name}")
    print(f"  Time: {datetime.utcnow().isoformat()}")
    print(f"{'='*60}\n")

    # ── Step 1: Check triggers ─────────────────────────────────────────────
    should_train, reason = _check_triggers(force)
    print(f"[RETRAIN] Trigger check: {reason}")
    if not should_train:
        print("[RETRAIN] Skipping — trigger conditions not met.")
        return {"outcome": "skipped", "reason": reason}

    if not OPENROUTER_API_KEY:
        print("[RETRAIN] ERROR: OPENROUTER_API_KEY not set — cannot run judge evaluation.")
        return {"outcome": "error", "reason": "OPENROUTER_API_KEY missing"}

    with mlflow.start_run(run_name=run_name) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("trigger_reason",  reason)
        mlflow.log_param("lora_r",          4)
        mlflow.log_param("lora_alpha",       8)
        mlflow.log_param("learning_rate",    1e-4)
        mlflow.log_param("model_name",       "patrickjohncyh/fashion-clip")
        mlflow.log_param("eval_method",      "judge_avg_20q_top5")
        # Logged so the metrics_exporter can show a progress bar in Grafana
        mlflow.log_param("max_train_steps",  int(os.getenv("MAX_TRAIN_STEPS", 300)))

        # ── Step 2: Build training pairs ──────────────────────────────────
        print("\n[RETRAIN] Step 2/5: Building training pairs...")
        from build_training_pairs import build_pairs
        manifest_path = build_pairs()
        n_pairs = sum(1 for _ in open(manifest_path)) - 2  # rough count
        mlflow.log_param("training_pairs", n_pairs)

        # ── Step 3: Train ─────────────────────────────────────────────────
        print("\n[RETRAIN] Step 3/5: Training LoRA adapters...")
        run_dir = RUNS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        from lora_trainer import train as train_lora
        adapter_path = train_lora(
            manifest_path  = str(manifest_path),
            output_dir     = str(run_dir),
            mlflow_run_id  = mlflow_run_id,
        )

        # ── Step 4: Evaluate via judge benchmark ──────────────────────────
        print("\n[RETRAIN] Step 4/5: Judge benchmark — old vs new model (20 queries × top-5)...")
        from promote_model import evaluate_with_judge, PROMOTION_DELTA
        old_avg, new_avg = evaluate_with_judge(
            new_adapter_dir            = adapter_path,
            new_visual_projection_path = str(run_dir / "visual_projection.pt"),
            api_key                    = OPENROUTER_API_KEY,
        )
        delta = new_avg - old_avg
        print(f"[RETRAIN] OLD avg={old_avg:.4f}  NEW avg={new_avg:.4f}  delta={delta:+.4f}")
        print(f"[RETRAIN] Promotion threshold: new_avg > old_avg + {PROMOTION_DELTA}")

        mlflow.log_metric("judge_old_avg", old_avg)
        mlflow.log_metric("judge_new_avg", new_avg)
        mlflow.log_metric("judge_delta",   delta)

        # ── Step 5: Promote or rollback ───────────────────────────────────
        print("\n[RETRAIN] Step 5/5: Promote / rollback decision...")
        promoted = new_avg > old_avg + PROMOTION_DELTA

        if promoted:
            from promote_model import promote_adapter
            promote_adapter(
                adapter_dir             = adapter_path,
                visual_projection_path  = str(run_dir / "visual_projection.pt"),
                mlflow_run_id           = mlflow_run_id,
            )
            mlflow.log_param("outcome", "promoted")
            mlflow.set_tag("promoted", "true")
            print(f"\n✅ PROMOTED — new adapter deployed (delta={delta:+.4f})")
        else:
            mlflow.log_param("outcome", "rejected")
            mlflow.set_tag("promoted", "false")
            print(f"\n❌ REJECTED — delta={delta:+.4f} does not exceed threshold +{PROMOTION_DELTA}")
            print(f"   Keeping current model. Adapter saved for inspection at: {adapter_path}")

        result = {
            "outcome":       "promoted" if promoted else "rejected",
            "judge_old_avg": old_avg,
            "judge_new_avg": new_avg,
            "judge_delta":   delta,
            "mlflow_run_id": mlflow_run_id,
            "adapter_path":  adapter_path,
        }

    print(f"\n{'='*60}")
    print(f"  Pipeline complete — outcome: {result['outcome']}")
    print(f"{'='*60}\n")
    return result


_LAST_RUN_FILE = SCRIPT_DIR / ".last_retrain_ts"


def _schedule_loop(interval_hours: float = 48.0):
    """Run the pipeline on a schedule, respecting the interval across container restarts."""
    interval_secs = interval_hours * 3600
    print(f"[RETRAIN] Scheduler started — running every {interval_hours}h")

    while True:
        # Check how long ago the last run finished (survives container restarts)
        now = time.time()
        if _LAST_RUN_FILE.exists():
            try:
                last_ts = float(_LAST_RUN_FILE.read_text().strip())
                elapsed = now - last_ts
                if elapsed < interval_secs:
                    wait = interval_secs - elapsed
                    print(
                        f"[RETRAIN] Last run was {elapsed/3600:.1f}h ago — "
                        f"next run in {wait/3600:.1f}h. Sleeping..."
                    )
                    time.sleep(wait)
                    continue
            except Exception:
                pass  # corrupt file — just run now

        try:
            result = run_pipeline(force=False)
            print(f"[RETRAIN] Scheduled run complete: {result['outcome']}")
        except Exception as e:
            print(f"[RETRAIN] Pipeline error: {e}")

        _LAST_RUN_FILE.write_text(str(time.time()))
        print(f"[RETRAIN] Next run in {interval_hours}h. Sleeping...")
        time.sleep(interval_secs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locus LoRA retraining pipeline")
    parser.add_argument("--force",    action="store_true", help="Skip trigger checks")
    parser.add_argument("--schedule", type=float, default=0,
                        help="Run on schedule every N hours (0 = run once and exit)")
    args = parser.parse_args()

    if args.schedule > 0:
        _schedule_loop(interval_hours=args.schedule)
    else:
        result = run_pipeline(force=args.force)
        sys.exit(0 if result["outcome"] != "error" else 1)
