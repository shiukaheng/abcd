#!/bin/bash
set -euo pipefail

cleanup() {
    echo ""
    echo "Benchmark interrupted by user"
    exit 1
}
trap cleanup INT TERM

# ────────────────────────────── Datasets ──────────────────────────────
DATASETS=("bonsai" "counter")

# ────────────────────────────── Configs ───────────────────────────────
# Fields: name | method | iterations | sync_interval | grid_size | min_gaussians
# Same as grid_smallb but with extra_cell_compensation=disabled
CONFIGS=(
    "grid_smallb_no_comp:grid_cpu:5000:1000:10:50"
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
        echo "[$current/$total] $dataset :: $cfg_name  ($method, iters=$iters, sync=$sync, grid=$grid, comp=disabled)"
        mkdir -p "$out_path"

        set +e
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
            --split-shrink-factor "$SPLIT_SHRINK_FACTOR" \
            --extra-cell-compensation disabled
        rc=$?
        set -e
        if [ $rc -ne 0 ]; then
            echo "  ⚠  FAILED (exit code $rc) — continuing"
        else
            echo "  → Generating plots..."
            python scripts/plot_graphs.py \
                --jsonl-path "${out_path}/benchmark.jsonl" \
                --output-dir "${out_path}/plots" || true
        fi
    done
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))

echo ""
echo "============================================================"
echo "Ablation benchmark complete!"
echo "Total time: ${hours}h ${minutes}m"
echo "Results saved to $OUTPUT_DIR"
echo "============================================================"