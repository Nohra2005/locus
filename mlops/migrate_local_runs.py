"""
migrate_local_runs.py — Imports real local experiment runs into the running MLflow server.

Reads from mlops/mlflow.db (local file-based tracking) and re-logs
the meaningful finished runs into the Docker-hosted server at localhost:5000.

Run once:
    python mlops/migrate_local_runs.py
"""
import sqlite3
import mlflow

LOCAL_DB    = "c:/dev/locus/mlops/mlflow.db"
REMOTE_URI  = "http://localhost:5000"

mlflow.set_tracking_uri(REMOTE_URI)

conn = sqlite3.connect(LOCAL_DB)
cur  = conn.cursor()


def get_metrics(run_uuid):
    cur.execute("SELECT key, value, step FROM metrics WHERE run_uuid=? ORDER BY key, step", (run_uuid,))
    rows = cur.fetchall()
    # Last value per key → scalar; multi-step keys → full history
    by_key = {}
    for k, v, step in rows:
        by_key.setdefault(k, []).append((step, float(v)))
    return by_key


def get_params(run_uuid):
    cur.execute("SELECT key, value FROM params WHERE run_uuid=?", (run_uuid,))
    return dict(cur.fetchall())


def log_run(experiment_name, run_name, run_uuid, extra_note=""):
    params  = get_params(run_uuid)
    metrics = get_metrics(run_uuid)

    if extra_note:
        params["note"] = extra_note

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        for k, v in params.items():
            mlflow.log_param(k, v)
        for key, history in metrics.items():
            for step, val in history:
                mlflow.log_metric(key, val, step=step)

    print(f"  ✓ {run_name}")


# ── Experiment 1: locus_visual_similarity ─────────────────────────────────────
# March 30 — establishing the cosine similarity baseline before any tuning.
# Only one unique result (two runs had identical metrics; keep the cleaner name).

print("\n[1] locus_visual_similarity")
log_run(
    "locus_visual_similarity",
    "cosine-similarity-baseline",
    "0dfd4da0e0124daf82dfdaf9a2e7b721",   # enthused-rook-536
    extra_note="Baseline: raw Fashion-CLIP cosine similarity on 30 golden queries before any prompt or YOLO tuning.",
)


# ── Experiment 2: locus_search_accuracy ───────────────────────────────────────
# April 3 — iterating on CLIP prompt templates + gateway category fallback.
# Three distinct result points; the last duplicate (sedate-wren-578) is skipped.

print("\n[2] locus_search_accuracy")
log_run(
    "locus_search_accuracy",
    "baseline-stale-crop",
    "0f17ae6441f942f7a7d33295811e2384",   # chill-lark-84
    extra_note="Baseline recall evaluation using stale bounding-box crops. hit@5=0.80.",
)
log_run(
    "locus_search_accuracy",
    "clip-prompts-v1-fix",
    "8f1a6b050f40454fb69b4a61bc5edd2a",   # clip_prompts_fix
    extra_note="Tweaked CLIP prompt templates (v1). hit@5 dropped to 0.73 — prompts overfit to label phrasing.",
)
log_run(
    "locus_search_accuracy",
    "clip-prompts-v2-plus-gateway-fallback",
    "1af49b28fb4b41a4b2ae82aa285067b0",   # clip_prompts_v2_plus_gateway_fallback
    extra_note="CLIP prompt v2 + added gateway category-fallback logic. hit@5 jumped to 0.967 — best result.",
)


# ── Experiment 3: locus_gemini_judge ──────────────────────────────────────────
# April 8-9 — switched evaluation metric to Gemini visual similarity score (vss5).
# Shows YOLO detector tuning and leggings category fix.

print("\n[3] locus_gemini_judge")
log_run(
    "locus_gemini_judge",
    "gemini-judge-initial",
    "d0482111e361435e9d70a1a4bbcf33fa",   # h (2026-04-07 21:54)
    extra_note="First working Gemini VSS evaluation. avg_vss5=0.635, avg_top1=0.801. Establishes judge baseline.",
)
log_run(
    "locus_gemini_judge",
    "yolo-world-baseline",
    "5a22bbd2b2564ae084a8ef8484bccfd9",   # yolo-world-baseline
    extra_note="YOLO-World detector integrated. avg_vss5=0.648. Per-category breakdown shows leggings worst at 0.37.",
)
log_run(
    "locus_gemini_judge",
    "yolo-world-v1-tier-analysis",
    "f90315e3c2a64934ab67a40b0a435956",   # yolo-world-v1-full-doc
    extra_note="Full tier analysis: T1-exact=28/30, T2-alias=1/30, fallback=1/30. Confirms YOLO crop strategy is sound.",
)
log_run(
    "locus_gemini_judge",
    "phase1-leggings-fix",
    "ef9369c0c5214a5585e5ee2300f16418",   # phase1-leggings-fix
    extra_note="Fixed leggings category alias in CLIP label mapping. cat/leggings_vss5: 0.37 → 0.712. avg_vss5=0.659.",
)
log_run(
    "locus_gemini_judge",
    "final-system-gt-vss",
    "4c25e35946df4a918e5687b01ed64512",   # gt_vss_complete (first of two)
    extra_note="Final evaluation: avg_vss5=0.659, avg_gt_vss=0.735, hit@5=1.0 on 30 golden queries. System accepted.",
)

conn.close()
print("\nMigration complete. Open http://localhost:5000")
