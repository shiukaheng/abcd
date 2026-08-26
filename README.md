# ABCD: Alpha-Composited Block Coordinate Descent

ABCD trains large 3D Gaussian Splatting scenes out of core by optimizing one
spatial partition at a time. Inactive partitions are stored on disk; their
cached renders provide the foreground and background context for the active
partition.

Paper: [SIGGRAPH Posters 2026](https://doi.org/10.1145/3799825.3818779)

![ABCD training example](images/demo.gif)

## Installation

Requirements: Linux, Python 3.11, a CUDA-capable NVIDIA GPU compatible with
PyTorch 2.3.1/CUDA 12.1, NVCC, a C++ compiler, and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
uv run python build_submodule.py submodules/diff-gaussian-rasterization
uv run python build_submodule.py submodules/simple-knn
```

The optional viewer is installed with `uv sync --frozen --extra viewer`.

## Dataset

The paper uses the Mip-NeRF 360 `garden` and `kitchen` scenes. Download the
resized dataset with:

```bash
./download_dataset.sh
```

Each scene must contain COLMAP sparse reconstruction files and an image
directory, for example:

```text
scene/
  images_4/
  sparse/0/cameras.bin
  sparse/0/images.bin
  sparse/0/points3D.bin
```

Every eighth camera is held out by default. Training records the exact camera
split in `run.json`; evaluation uses that same split.

## Training

```bash
uv run abcd-train \
  --dataset datasets/garden \
  --output results/garden-abcd \
  --method abcd \
  --partition-size 5 \
  --iterations 5000 \
  --sync-interval 250 \
  --seed 0
```

Supported methods are:

- `abcd`: disk-backed alpha compositing.
- `abcd-no-compositing`: the partitioned ablation.
- `3dgs`: the unpartitioned baseline.

Runs are headless by default and write `run.json`, `training.jsonl`,
`model.ply`, and an ABCD `cache/` directory. Cached RGB and alpha use `uint8`;
depth uses `float16`. Cache and shard writes are atomic and checksum-verified.
Interrupted ABCD runs resume from the shard and optimizer checkpoints when
restarted with the same configuration.

## Evaluation

```bash
uv run abcd-evaluate \
  --model results/garden-abcd/model.ply \
  --dataset datasets/garden \
  --output results/garden-abcd/evaluation.csv
```

Evaluation reports held-out per-view PSNR, SSIM, and LPIPS.

## Reproduction

Run the paper configuration for all methods and scenes:

```bash
uv run abcd-reproduce \
  --dataset-root datasets \
  --output results/siggraph_2026
```

The command reuses completed models and evaluations unless `--force` is given.
It produces `results.json`, `results.csv`, and `comparison.png` in the output
directory. The function defaults are visible in `src/abcd/reproduce.py`; pass
arguments such as `--iterations 10000` to override them.

Equivalent runnable entrypoints live in `scripts/`, for example:

```bash
uv run python scripts/reproduce.py --dataset-root datasets --output results/paper
```

## Memory Model

ABCD has three storage tiers:

```text
disk -> inactive partition checkpoints and cached renders
RAM  -> active partition context and camera data
VRAM -> one active partition, optimizer state, and one camera working set
```

For fixed image resolution, fixed partition volume, and bounded expected
Gaussian density, peak training VRAM is `O(G_cell + H * W)`, where `G_cell` is
the active partition's Gaussian count. This claim does not cover disk use,
system RAM, camera count, runtime, or cells with unbounded density.

Measure inactive-shard VRAM scaling with:

```bash
uv run abcd-memory --output release_checks/memory.json
```

The command holds active-shard size fixed across 1, 2, 4, and 8 partitions and
fails when peak allocated VRAM spreads by 1 MiB or more.

## Tests

```bash
uv sync --frozen --group dev
uv run ruff format --check src tests scripts build_submodule.py
uv run ruff check src tests scripts build_submodule.py
uv run pyright
uv run pytest
```

## Limitations

- Gaussian ownership is assigned by mean, so covariance support may cross a
  partition boundary.
- Partition-center depth is an approximation for views where partitions cannot
  be globally ordered.
- The implementation targets one CUDA GPU.

## License

Original ABCD code is released under the MIT License. Bundled GraphDECO
rasterizer and simple-knn code have separate non-commercial research and
evaluation terms. See `LICENSE`, `THIRD_PARTY_NOTICES.md`, and the licenses in
`submodules/`.

## Citation

```bibtex
@inproceedings{shiu2026abcd,
  title     = {ABCD: Alpha-Composited Block Coordinate Descent},
  author    = {Shiu, Ka Heng and Subr, Kartic},
  booktitle = {SIGGRAPH Posters},
  year      = {2026},
  doi       = {10.1145/3799825.3818779}
}
```
