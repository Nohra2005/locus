# -*- coding: utf-8 -*-
"""
Locus — Gemini Judge Metrics Exporter
======================================
Reads gemini_judge_results.json and vss_cache.json every 30s and
exposes them as Prometheus metrics on port 8003.

Runs as a Docker service so Grafana can display Gemini quality metrics
alongside the gateway's operational metrics.

Metrics exposed:
  locus_gemini_avg_vss5               — average VSS@5 across all evaluated queries
  locus_gemini_query_vss5{label}      — VSS@5 per individual query
  locus_gemini_cache_entries          — total entries in vss_cache.json
  locus_gemini_queries_evaluated      — how many queries were in the last eval run
  locus_gemini_queries_poor           — queries with VSS@5 < 0.55
  locus_gemini_queries_fair           — queries with VSS@5 in [0.55, 0.70)
  locus_gemini_queries_good           — queries with VSS@5 in [0.70, 0.85)
  locus_gemini_queries_excellent      — queries with VSS@5 >= 0.85
  locus_gemini_score_bucket{le}       — score distribution histogram buckets
"""

import json
import os
import time

from prometheus_client import Gauge, start_http_server

# ── File paths (overridable via env) ─────────────────────────────────────────
RESULTS_PATH   = os.getenv("GEMINI_RESULTS_PATH", "/mlops/gemini_judge_results.json")
CACHE_PATH     = os.getenv("VSS_CACHE_PATH",       "/mlops/vss_cache.json")
EXPORT_PORT    = int(os.getenv("METRICS_PORT",      "8003"))
REFRESH_SECS   = int(os.getenv("REFRESH_SECS",      "30"))

# ── Prometheus metrics ────────────────────────────────────────────────────────

avg_vss5          = Gauge("locus_gemini_avg_vss5",          "Average VSS@5 across all queries in latest eval run")
query_vss5        = Gauge("locus_gemini_query_vss5",         "VSS@5 score per query", ["label"])
cache_entries     = Gauge("locus_gemini_cache_entries",      "Total entries in vss_cache.json")
queries_evaluated = Gauge("locus_gemini_queries_evaluated",  "Queries evaluated in latest run")
queries_poor      = Gauge("locus_gemini_queries_poor",       "Queries with VSS@5 < 0.55")
queries_fair      = Gauge("locus_gemini_queries_fair",       "Queries with VSS@5 in [0.55, 0.70)")
queries_good      = Gauge("locus_gemini_queries_good",       "Queries with VSS@5 in [0.70, 0.85)")
queries_excellent = Gauge("locus_gemini_queries_excellent",  "Queries with VSS@5 >= 0.85")
top1_avg          = Gauge("locus_gemini_avg_top1",           "Average Top-1 VSS score across all queries")

# Score distribution buckets (single Gauge, differentiated by 'le' label)
score_bucket_gauge = Gauge(
    "locus_gemini_score_bucket",
    "Count of individual result scores in this range",
    ["le"],
)


def refresh():
    """Read JSON files and update all gauges."""

    # ── Gemini judge results ──────────────────────────────────────────────────
    if not os.path.exists(RESULTS_PATH):
        print(f"[exporter] Results file not found: {RESULTS_PATH} — waiting for first eval run")
        return

    try:
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    except Exception as e:
        print(f"[exporter] Could not read results file: {e}")
        return

    vss5_list  = [r["vss5"]       for r in results if "vss5"       in r]
    top1_list  = [r["top1_score"] for r in results if "top1_score" in r]
    all_scores = [s for r in results for s in r.get("scores", [])]

    if not vss5_list:
        return

    # Aggregate metrics
    avg = sum(vss5_list) / len(vss5_list)
    avg_vss5.set(round(avg, 4))
    queries_evaluated.set(len(vss5_list))

    if top1_list:
        top1_avg.set(round(sum(top1_list) / len(top1_list), 4))

    # Grade distribution
    queries_poor.set(      sum(1 for s in vss5_list if s < 0.55))
    queries_fair.set(      sum(1 for s in vss5_list if 0.55 <= s < 0.70))
    queries_good.set(      sum(1 for s in vss5_list if 0.70 <= s < 0.85))
    queries_excellent.set( sum(1 for s in vss5_list if s >= 0.85))

    # Per-query metrics
    for r in results:
        if "vss5" in r:
            label = r["label"][:50]   # Prometheus label max ~64 chars
            query_vss5.labels(label=label).set(r["vss5"])

    # Score distribution buckets
    for threshold in [0.2, 0.4, 0.6, 0.8, 1.0]:
        count = sum(1 for s in all_scores if s <= threshold)
        score_bucket_gauge.labels(le=str(threshold)).set(count)

    print(
        f"[exporter] Refreshed — {len(vss5_list)} queries, "
        f"avg_vss5={avg:.3f}, "
        f"poor={int(queries_poor._value.get())} "
        f"fair={int(queries_fair._value.get())} "
        f"good={int(queries_good._value.get())} "
        f"excellent={int(queries_excellent._value.get())}"
    )

    # ── VSS cache ─────────────────────────────────────────────────────────────
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                cache = json.load(f)
            cache_entries.set(len(cache))
        except Exception as e:
            print(f"[exporter] Could not read cache file: {e}")


if __name__ == "__main__":
    print(f"[exporter] Starting Locus Gemini metrics exporter on port {EXPORT_PORT}")
    print(f"[exporter] Results path : {RESULTS_PATH}")
    print(f"[exporter] Cache path   : {CACHE_PATH}")
    print(f"[exporter] Refresh every: {REFRESH_SECS}s")

    start_http_server(EXPORT_PORT)
    print(f"[exporter] Prometheus metrics available at http://0.0.0.0:{EXPORT_PORT}/metrics")

    while True:
        refresh()
        time.sleep(REFRESH_SECS)
