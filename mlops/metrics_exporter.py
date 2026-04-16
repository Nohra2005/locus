# -*- coding: utf-8 -*-
"""
Locus — MLops Metrics Exporter
================================
Reads evaluation results from MLflow AND Gemini judge JSON files every 30s,
then exposes all of them as Prometheus metrics on port 8003.

MLflow-based metrics (LoRA retraining pipeline):
  locus_recall_at_5               — latest Recall@5 from locus_search_accuracy
  locus_recall_at_10              — latest Recall@10
  locus_recall_at_25              — latest Recall@25
  locus_baseline_hit5             — best hit@5 baseline (for promotion decisions)
  locus_lora_delta_hit5           — delta hit@5 from latest retraining run
  locus_lora_promoted_total       — cumulative adapter promotions
  locus_lora_runs_total           — total retraining pipeline runs
  locus_queries_evaluated_total   — number of golden queries in latest eval run
  locus_retrain_active            — 1 if a retrain run is currently RUNNING, else 0
  locus_retrain_step              — latest logged training step in the active run
  locus_lora_train_loss           — latest train_loss value in the active run
  locus_retrain_max_steps         — MAX_TRAIN_STEPS param from the active run

Gemini Judge metrics (from gemini_judge_results.json):
  locus_gemini_avg_vss5           — average VSS@5 across all evaluated queries
  locus_gemini_query_vss5{label}  — VSS@5 per individual query
  locus_gemini_cache_entries      — total entries in vss_cache.json
  locus_gemini_queries_evaluated  — how many queries were in the last eval run
  locus_gemini_queries_poor/fair/good/excellent — grade distribution
  locus_gemini_avg_top1           — average Top-1 VSS score
  locus_gemini_score_bucket{le}   — score distribution histogram buckets
"""

import json
import os
import time

from prometheus_client import Gauge, Counter, start_http_server

MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPORT_PORT    = int(os.getenv("METRICS_PORT", "8003"))
REFRESH_SECS   = int(os.getenv("REFRESH_SECS", "30"))
RESULTS_PATH   = os.getenv("GEMINI_RESULTS_PATH", "/mlops/gemini_judge_results.json")
CACHE_PATH     = os.getenv("VSS_CACHE_PATH",       "/mlops/vss_cache.json")

# ── Prometheus metrics ────────────────────────────────────────────────────────

recall_at_5  = Gauge("locus_recall_at_5",  "Latest Recall@5 on golden dataset")
recall_at_10 = Gauge("locus_recall_at_10", "Latest Recall@10 on golden dataset")
recall_at_25 = Gauge("locus_recall_at_25", "Latest Recall@25 on golden dataset")

baseline_hit5    = Gauge("locus_baseline_hit5",    "Best hit@5 baseline used for promotion decisions")
lora_delta_hit5  = Gauge("locus_lora_delta_hit5",  "hit@5 delta in latest retraining run (new - baseline)")

lora_promoted = Gauge("locus_lora_promoted_total", "Cumulative adapter promotions")
lora_runs     = Gauge("locus_lora_runs_total",      "Total retraining pipeline runs (promoted + rejected)")
queries_eval  = Gauge("locus_queries_evaluated_total", "Golden queries evaluated in latest recall run")

# Live training metrics — updated whenever a run is RUNNING
retrain_active    = Gauge("locus_retrain_active",     "1 if a retrain run is currently active, else 0")
retrain_step      = Gauge("locus_retrain_step",       "Latest logged training step in the active run")
lora_train_loss   = Gauge("locus_lora_train_loss",    "Latest train_loss value logged by the active run")
retrain_max_steps = Gauge("locus_retrain_max_steps",  "MAX_TRAIN_STEPS param of the active run")

# Latest completed run evaluation results
lora_new_hit5  = Gauge("locus_lora_new_hit5",  "hit@5 of the newly trained adapter in the last retrain run")
lora_new_acs5  = Gauge("locus_lora_new_acs5",  "ACS@5 of the newly trained adapter in the last retrain run")

# ── Gemini Judge metrics ──────────────────────────────────────────────────────
gemini_avg_vss5      = Gauge("locus_gemini_avg_vss5",         "Average VSS@5 across all queries in latest eval run")
gemini_query_vss5    = Gauge("locus_gemini_query_vss5",        "VSS@5 score per query", ["label"])
gemini_cache_entries = Gauge("locus_gemini_cache_entries",     "Total entries in vss_cache.json")
gemini_queries_eval  = Gauge("locus_gemini_queries_evaluated", "Queries evaluated in latest Gemini run")
gemini_queries_poor  = Gauge("locus_gemini_queries_poor",      "Queries with VSS@5 < 0.55")
gemini_queries_fair  = Gauge("locus_gemini_queries_fair",      "Queries with VSS@5 in [0.55, 0.70)")
gemini_queries_good  = Gauge("locus_gemini_queries_good",      "Queries with VSS@5 in [0.70, 0.85)")
gemini_queries_excel = Gauge("locus_gemini_queries_excellent", "Queries with VSS@5 >= 0.85")
gemini_top1_avg      = Gauge("locus_gemini_avg_top1",          "Average Top-1 VSS score across all queries")
gemini_score_bucket  = Gauge("locus_gemini_score_bucket",      "Count of result scores <= threshold", ["le"])


def _get_mlflow_client():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.tracking.MlflowClient()


def _refresh_recall_metrics(client) -> None:
    """Pull latest Recall@K from 'locus_search_accuracy' experiment."""
    try:
        exp = client.get_experiment_by_name("locus_search_accuracy")
        if exp is None:
            return

        runs = client.search_runs(
            experiment_ids = [exp.experiment_id],
            order_by       = ["start_time DESC"],
            max_results    = 1,
        )
        if not runs:
            return

        metrics = runs[0].data.metrics
        if "recall_at_5" in metrics:
            recall_at_5.set(round(metrics["recall_at_5"], 4))
        if "recall_at_10" in metrics:
            recall_at_10.set(round(metrics["recall_at_10"], 4))
        if "recall_at_25" in metrics:
            recall_at_25.set(round(metrics["recall_at_25"], 4))
        if "queries_evaluated" in metrics:
            queries_eval.set(int(metrics["queries_evaluated"]))

        print(
            f"[exporter] Recall — "
            f"R@5={metrics.get('recall_at_5', '?'):.3f}  "
            f"R@10={metrics.get('recall_at_10', '?'):.3f}  "
            f"R@25={metrics.get('recall_at_25', '?'):.3f}  "
            f"n={int(metrics.get('queries_evaluated', 0))}"
        )

    except Exception as e:
        print(f"[exporter] Could not read locus_search_accuracy from MLflow: {e}")


def _refresh_retrain_metrics(client) -> None:
    """Pull retraining stats from 'locus_lora_retraining' experiment."""
    try:
        exp = client.get_experiment_by_name("locus_lora_retraining")
        if exp is None:
            return

        # All runs for aggregate counts
        all_runs = client.search_runs(
            experiment_ids = [exp.experiment_id],
            max_results    = 1000,
        )
        if not all_runs:
            return

        lora_runs.set(len(all_runs))

        promoted_count = sum(
            1 for r in all_runs
            if r.data.tags.get("promoted") == "true"
        )
        lora_promoted.set(promoted_count)

        # Latest run for delta
        latest = sorted(all_runs, key=lambda r: r.info.start_time, reverse=True)[0]
        delta  = latest.data.metrics.get("delta_hit5")
        bline  = latest.data.params.get("baseline_hit5")

        if delta is not None:
            lora_delta_hit5.set(round(float(delta), 4))
        if bline is not None:
            baseline_hit5.set(round(float(bline), 4))

        new_h5 = latest.data.metrics.get("new_hit5")
        new_a5 = latest.data.metrics.get("new_acs5")
        if new_h5 is not None:
            lora_new_hit5.set(round(float(new_h5), 4))
        if new_a5 is not None:
            lora_new_acs5.set(round(float(new_a5), 4))

        outcome = latest.data.params.get("outcome", "unknown")
        print(
            f"[exporter] Retrain — runs={len(all_runs)} promoted={promoted_count} "
            f"latest_outcome={outcome} delta_hit5={delta} "
            f"new_hit5={new_h5} new_acs5={new_a5}"
        )

    except Exception as e:
        print(f"[exporter] Could not read locus_lora_retraining from MLflow: {e}")


def _refresh_live_training_metrics(client) -> None:
    """
    Poll for any RUNNING locus_lora_retraining run and export its live metrics.

    MLflow writes log_metric() calls synchronously to the backend, so the
    latest train_loss from an in-progress run is visible here between steps.
    When no run is active, all live gauges are zeroed.
    """
    try:
        exp = client.get_experiment_by_name("locus_lora_retraining")
        if exp is None:
            retrain_active.set(0)
            return

        running = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not running:
            retrain_active.set(0)
            retrain_step.set(0)
            lora_train_loss.set(0)
            print("[exporter] Live training: no active run")
            return

        run = running[0]
        retrain_active.set(1)

        # Latest train_loss — get_metric_history returns all logged points;
        # the last one has the highest step.
        try:
            history = client.get_metric_history(run.info.run_id, "train_loss")
            if history:
                latest = max(history, key=lambda m: m.step)
                lora_train_loss.set(round(latest.value, 6))
                retrain_step.set(latest.step)
            else:
                # Run started but hasn't logged a step yet (still in pair-build/init phase)
                lora_train_loss.set(0)
                retrain_step.set(0)
        except Exception as e:
            print(f"[exporter] Could not read train_loss history: {e}")

        # MAX_TRAIN_STEPS stored as a run param
        try:
            max_s = run.data.params.get("max_train_steps")
            if max_s is not None:
                retrain_max_steps.set(int(max_s))
        except Exception:
            pass

        print(
            f"[exporter] Live training — run={run.info.run_id[:8]} "
            f"step={int(retrain_step._value.get())} "
            f"loss={lora_train_loss._value.get():.4f}"
        )

    except Exception as e:
        print(f"[exporter] Could not read live training metrics: {e}")
        retrain_active.set(0)


def _refresh_gemini_metrics() -> None:
    """Read gemini_judge_results.json and vss_cache.json, update Gemini gauges."""
    if not os.path.exists(RESULTS_PATH):
        print(f"[exporter] Gemini results not found: {RESULTS_PATH} — waiting for first eval run")
        return
    try:
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    except Exception as e:
        print(f"[exporter] Could not read Gemini results: {e}")
        return

    vss5_list  = [r["vss5"]       for r in results if "vss5"       in r]
    top1_list  = [r["top1_score"] for r in results if "top1_score" in r]
    all_scores = [s for r in results for s in r.get("scores", [])]

    if not vss5_list:
        return

    avg = sum(vss5_list) / len(vss5_list)
    gemini_avg_vss5.set(round(avg, 4))
    gemini_queries_eval.set(len(vss5_list))
    if top1_list:
        gemini_top1_avg.set(round(sum(top1_list) / len(top1_list), 4))

    gemini_queries_poor.set( sum(1 for s in vss5_list if s < 0.55))
    gemini_queries_fair.set( sum(1 for s in vss5_list if 0.55 <= s < 0.70))
    gemini_queries_good.set( sum(1 for s in vss5_list if 0.70 <= s < 0.85))
    gemini_queries_excel.set(sum(1 for s in vss5_list if s >= 0.85))

    for r in results:
        if "vss5" in r:
            gemini_query_vss5.labels(label=r["label"][:50]).set(r["vss5"])

    for threshold in [0.2, 0.4, 0.6, 0.8, 1.0]:
        gemini_score_bucket.labels(le=str(threshold)).set(
            sum(1 for s in all_scores if s <= threshold)
        )

    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                cache = json.load(f)
            gemini_cache_entries.set(len(cache))
        except Exception as e:
            print(f"[exporter] Could not read VSS cache: {e}")

    print(
        f"[exporter] Gemini — {len(vss5_list)} queries, avg_vss5={avg:.3f}, "
        f"poor={int(gemini_queries_poor._value.get())} "
        f"fair={int(gemini_queries_fair._value.get())} "
        f"good={int(gemini_queries_good._value.get())} "
        f"excellent={int(gemini_queries_excel._value.get())}"
    )


def refresh() -> None:
    try:
        client = _get_mlflow_client()
        _refresh_recall_metrics(client)
        _refresh_retrain_metrics(client)
        _refresh_live_training_metrics(client)
    except Exception as e:
        print(f"[exporter] MLflow connection error: {e}")
    _refresh_gemini_metrics()


if __name__ == "__main__":
    print(f"[exporter] Starting Locus MLops metrics exporter on port {EXPORT_PORT}")
    print(f"[exporter] MLflow URI    : {MLFLOW_URI}")
    print(f"[exporter] Refresh every : {REFRESH_SECS}s")

    start_http_server(EXPORT_PORT)
    print(f"[exporter] Prometheus metrics at http://0.0.0.0:{EXPORT_PORT}/metrics")

    while True:
        refresh()
        time.sleep(REFRESH_SECS)
