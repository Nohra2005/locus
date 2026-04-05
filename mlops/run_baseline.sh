#!/bin/bash
# run_baseline.sh — Run a clean MLflow evaluation against the current model.
#
# Usage:
#   cd mlops/
#   bash run_baseline.sh
#
# This script activates the mlops venv, waits for services to be healthy,
# then runs evaluate_mlflow.py with the run name pre-set so you don't have
# to type it interactively.
#
# Requires: docker compose up (gateway on :8000, mlflow on :5000)

set -e

GATEWAY="http://localhost:8000"
MLFLOW="http://localhost:5000"
RUN_NAME="${1:-fashion_clip_baseline}"

echo "=============================================="
echo "  Locus Baseline Evaluation"
echo "  Run name : $RUN_NAME"
echo "  Gateway  : $GATEWAY"
echo "  MLflow   : $MLFLOW"
echo "=============================================="

# ── Check services are up ─────────────────────────────────────────────────────
echo ""
echo "Checking services..."

for i in {1..5}; do
    if curl -sf "$GATEWAY/" > /dev/null 2>&1; then
        echo "  ✅ Gateway is up"
        break
    fi
    if [ $i -eq 5 ]; then
        echo "  ❌ Gateway not reachable at $GATEWAY"
        echo "     Make sure docker compose is running: docker compose up -d"
        exit 1
    fi
    echo "  ⏳ Waiting for gateway... ($i/5)"
    sleep 3
done

for i in {1..5}; do
    if curl -sf "$MLFLOW/" > /dev/null 2>&1; then
        echo "  ✅ MLflow is up"
        break
    fi
    if [ $i -eq 5 ]; then
        echo "  ❌ MLflow not reachable at $MLFLOW"
        echo "     Make sure docker compose is running: docker compose up -d"
        exit 1
    fi
    echo "  ⏳ Waiting for MLflow... ($i/5)"
    sleep 3
done

# ── Activate venv and run evaluation ─────────────────────────────────────────
echo ""
echo "Activating mlops venv..."
source "$(dirname "$0")/venv/bin/activate"

echo "Running evaluation (run name: '$RUN_NAME')..."
echo ""

# Pass run name via stdin to skip the interactive prompt
echo "$RUN_NAME" | python3 "$(dirname "$0")/evaluate_mlflow.py"

echo ""
echo "Done. View results at: $MLFLOW"
