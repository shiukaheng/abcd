#!/bin/bash
set -euo pipefail

cleanup() {
    echo ""
    echo "Benchmark interrupted by user"
    exit 1
}
trap cleanup INT TERM

# ────────────────────────────── Datasets ──────────────────────────────
DATASETS=("bicycle" "bonsai" "counter" "garden")

# ────────────────────────────── Configs ───────────────────────────────
# Fields: name | method | iterations | sync_interval | grid_size | min_gaussians
#   name          – human-readable label for output directory
#   method        – "vanilla" or "grid_cpu"
#   iterations    – total iterations (per cell for grid)
#   sync_interval – grid only; interval between cell syncs
#   grid_size     – grid only; side length of each spatial cell
#   min_gaussians – grid only; minimum gaussians per cell
CONFIGS=(
    "vanilla:vanilla:5000:1000:50:50"
    "grid_og:grid_cpu:5000:1000:50:50"
    "grid_small:grid_cpu:5000:1000:25:50"
    "grid_shortsync:grid_cpu:5000:500:50:50"
)

# ─────────────────────── Shared hyperparameters ───────────────────────
IMAGES_SUBDIR="images_4"
OUTPUT_DIR="./benchmark_results"

DENSIFY_INTERVAL=100
DENSIFY_FROM_ITER=500
DENSIFY_UNTIL_ITER=15000
DENSIFY_GRAD_THRESHOLD=0.0002
OPACITY_THRESHOLD=0.005
SPLIT_N_SAMPLES=2
SPLIT_SHRINK_FACTOR=0.8

# ─────────────────────────────── Main ─────────────────────────────────
total=$((${#DATASETS[@]} * ${#CONFIGS[@]}))
current=0
start_time=$(date +%s)

mkdir -p "$OUTPUT_DIR"

for dataset in "${DATASETS[@]}"; do
    for config_tuple in "${CONFIGS[@]}"; do
        IFS=':' read -r cfg_name method iters sync grid min_g <<< "$config_tuple"
        current=$((current + 1))
        out_path="${OUTPUT_DIR}/${dataset}_${cfg_name}"

        echo ""
        echo "[$current/$total] $dataset :: $cfg_name  ($method, iters=$iters, sync=$sync, grid=$grid)"
        mkdir -p "$out_path"

        if [ "$method" = "vanilla" ]; then
            uv run python run_benchmark.py \
                --dataset "./datasets/$dataset" \
                --output "$out_path" \
                --method "$method" \
                --iterations "$iters" \
                --images-subdir "$IMAGES_SUBDIR" \
                --densify-interval "$DENSIFY_INTERVAL" \
                --densify-from-iter "$DENSIFY_FROM_ITER" \
                --densify-until-iter "$DENSIFY_UNTIL_ITER" \
                --densify-grad-threshold "$DENSIFY_GRAD_THRESHOLD" \
                --opacity-threshold "$OPACITY_THRESHOLD" \
                --split-n-samples "$SPLIT_N_SAMPLES" \
                --split-shrink-factor "$SPLIT_SHRINK_FACTOR"
        else
            uv run python run_benchmark.py \
                --dataset "./datasets/$dataset" \
                --output "$out_path" \
                --method "$method" \
                --iterations "$iters" \
                --sync-interval "$sync" \
                --grid-size "$grid" \
                --min-gaussians "$min_g" \
                --images-subdir "$IMAGES_SUBDIR" \
                --densify-interval "$DENSIFY_INTERVAL" \
                --densify-from-iter "$DENSIFY_FROM_ITER" \
                --densify-until-iter "$DENSIFY_UNTIL_ITER" \
                --densify-grad-threshold "$DENSIFY_GRAD_THRESHOLD" \
                --opacity-threshold "$OPACITY_THRESHOLD" \
                --split-n-samples "$SPLIT_N_SAMPLES" \
                --split-shrink-factor "$SPLIT_SHRINK_FACTOR"
        fi
    done
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))

echo ""
echo "============================================================"
echo "Benchmark complete!"
echo "Total time: ${hours}h ${minutes}m"
echo "Results saved to $OUTPUT_DIR"
echo "============================================================"
