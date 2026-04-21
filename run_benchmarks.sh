#!/bin/bash

DATASETS=("bonsai" "kitchen" "garden" "room" "stump")
METHODS=("vanilla" "grid_naive" "grid_gpu" "grid_cpu")

GRID_SIZE=5
ITERATIONS=5000
SYNC_INTERVAL=250
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
        
        python run_benchmark.py \
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
