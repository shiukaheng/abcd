#!/bin/bash

set -e

cleanup() {
    echo ""
    echo "Benchmark interrupted by user"
    exit 1
}

trap cleanup INT TERM

DATASETS=("bonsai" "kitchen" "garden" "room" "stump")
METHODS=("vanilla" "grid_naive" "grid_gpu" "grid_cpu")
ITERATIONS=5000
SYNC_INTERVAL=250
SYNC_INTERVAL=100
MIN_GAUSSIANS=50
IMAGES_SUBDIR="images_4"
OUTPUT_DIR="./benchmark_results"

total=$((${#DATASETS[@]} * ${#METHODS[@]}))
current=0
start_time=$(date +%s)

mkdir -p "$OUTPUT_DIR"

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        current=$((current + 1))
        echo ""
        echo "[$current/$total] Starting $dataset / $method"
        
        uv run run_benchmark.py \
            --dataset "./datasets/$dataset" \
            --output "$OUTPUT_DIR/${dataset}_${method}" \
            --method "$method" \
            --iterations "$ITERATIONS" \
            --grid-size "$GRID_SIZE" \
            --sync-interval "$SYNC_INTERVAL" \
            --min-gaussians "$MIN_GAUSSIANS" \
            --images-subdir "$IMAGES_SUBDIR"
    done
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))

echo ""
echo "============================================================"
echo "Benchmark complete!"
echo "Total time: $hours hours"
echo "Results saved to $OUTPUT_DIR"
echo "============================================================"

echo ""
echo "Generating visualizations..."
uv run visualize_benchmark.py --benchmark-dir "$OUTPUT_DIR" --output-dir "./benchmark_plots"
