#!/usr/bin/env bash
set -euo pipefail

# POC test: vanilla vs grid_cpu on kitchen and counter datasets.
# Runs both methods, converts outputs to Chrome trace format for Perfetto.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

DATASETS=("kitchen")  # "counter"
METHODS=("vanilla" "grid_cpu")
ITERATIONS=5000
SYNC_INTERVAL=1000
GRID_SIZE=50
MIN_GAUSSIANS=100

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        OUTPUT_DIR="./new_eval/${dataset}/${method}"
        mkdir -p "$OUTPUT_DIR"

        echo ""
        echo "============================================================"
        echo "Dataset: ${dataset}  |  Method: ${method}"
        echo "Output:  ${OUTPUT_DIR}"
        echo "============================================================"

        if [ "$method" = "vanilla" ]; then
            uv run python run_benchmark.py \
                --dataset "./datasets/${dataset}" \
                --output "$OUTPUT_DIR" \
                --method vanilla \
                --iterations "$ITERATIONS"
        elif [ "$method" = "grid_cpu" ]; then
            uv run python run_benchmark.py \
                --dataset "./datasets/${dataset}" \
                --output "$OUTPUT_DIR" \
                --method grid_cpu \
                --iterations "$ITERATIONS" \
                --sync-interval "$SYNC_INTERVAL" \
                --grid-size "$GRID_SIZE" \
                --min-gaussians "$MIN_GAUSSIANS"
        fi

        echo "Converting to trace..."
        python scripts/jsonl_to_trace.py \
            "${OUTPUT_DIR}/benchmark.jsonl" \
            -o "${OUTPUT_DIR}/trace.json"

        echo "Done: ${OUTPUT_DIR}/trace.json"
    done
done

echo ""
echo "============================================================"
echo "All done. Drag these into https://ui.perfetto.dev:"
echo "============================================================"
for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        echo "  ./new_eval/${dataset}/${method}/trace.json"
    done
done
