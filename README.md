# ABCD: Alpha-Composited Block Coordinate Descent

ABCD trains large 3D Gaussian Splatting scenes out of core by optimizing one
spatial partition at a time. Inactive partitions are stored on disk; their
cached renders provide the foreground and background context for the active
partition.

Paper: [SIGGRAPH Posters 2026](https://doi.org/10.1145/3799825.3818779)
[Video](https://youtu.be/3jBalwElgFM)

![ABCD training example](images/demo.gif)

## See ABCD Live

The most direct way to understand ABCD is to watch it train. This command opens
two live views while it optimizes spatial partitions:

```bash
uv sync --frozen
uv run python scripts/train.py \
  --dataset datasets/garden \
  --output results/garden-live \
  --method abcd \
  --partition-size 40 \
  --iterations 5000 \
  --preview 0

# --dataset datasets/garden: input COLMAP scene.
# --output results/garden-live: checkpoints, cache, logs, and final model.
# --method abcd: train one spatial block at a time while the rest stays visible.
# --partition-size 40: use roughly 11 large blocks for a legible garden demo.
# --iterations 5000: give each block 5,000 image-matching updates.
# --preview 0: keep the OpenCV window on camera 0 to compare changes over time.
```

Open <http://localhost:8080> in a browser. The web viewer shows the block being
optimized, its boundary, all scene cameras and reference images, and a freely
movable live render. The OpenCV window stays on one reference camera. Together
they make the ABCD loop visible: the browser reveals which local block is being
updated, while the OpenCV view shows how that update improves the full scene.

## Installation

Requirements: Linux, Python 3.11, a CUDA-capable NVIDIA GPU compatible with
PyTorch 2.3.1/CUDA 12.1, NVCC, a C++ compiler, and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
uv run python build_native.py vendor/diff-gaussian-rasterization
uv run python build_native.py vendor/simple-knn
```

`uv sync` also installs the web viewer used during interactive training.

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

# --dataset datasets/garden: input COLMAP scene.
# --output results/garden-abcd: checkpoints, cache, logs, and final model.
# --method abcd: train one spatial block at a time while the rest stays visible.
# --partition-size 5: split the scene into cubes with five-unit edges.
# --iterations 5000: give each block 5,000 image-matching updates.
# --sync-interval 250: switch to another block after every 250 updates.
# --seed 0: reproducible camera order and initialization.
```

Supported methods are:

- `abcd`: disk-backed alpha compositing.
- `abcd-no-compositing`: the partitioned ablation.
- `3dgs`: the unpartitioned baseline.

Runs write `run.json`, `training.jsonl`, `model.ply`, and an ABCD `cache/`
directory. Cached RGB and alpha use `uint8`; depth uses `float16`. Cache and
shard writes are atomic and checksum-verified.
Interrupted ABCD runs resume from the shard and optimizer checkpoints when
restarted with `--resume` and the same configuration. Every run otherwise
discards the existing cache in its output directory before starting fresh.

`--cache-storage disk` is the default: it bounds RAM use and supports explicit
resume. On a machine with enough RAM, `--cache-storage ram` keeps inactive
partition models and cached renders in host memory for faster block switches.
RAM cache is cleared when the process exits and cannot resume.

## Block Switching

ABCD switches to another spatial block every `--sync-interval` updates. With
`--cache-storage disk`, a switch reads the next block's model and cached renders
from disk, so frequent switching can become I/O-bound. Increase
`--sync-interval` to switch less often, or use `--cache-storage ram` when the
machine has enough host memory to keep all blocks and cached renders resident.
RAM mode is faster but is not resumable and scales with total scene size.

## Interactive Viewer

Training starts a Viser server at <http://localhost:8080> by default. Open it
in a browser while training to inspect the block being optimized, its
highlighted boundary, all scene cameras, and the live renderer output.
Move the browser camera independently while training continues; use the render
channel control to switch between RGB, depth, and alpha.

For a second, fixed training-camera render in an OpenCV window, add
`--preview 0` or `--preview all`. Disable the browser viewer for batch runs
with `--headless`.

## Evaluation

```bash
uv run abcd-evaluate \
  --model results/garden-abcd/model.ply \
  --dataset datasets/garden \
  --output results/garden-abcd/evaluation.csv

# --model results/garden-abcd/model.ply: final PLY written by training.
# --dataset datasets/garden: the same COLMAP scene used for training.
# --output results/garden-abcd/evaluation.csv: per-view held-out metrics.
```

Evaluation reports held-out per-view PSNR, SSIM, and LPIPS.

## Reproduction

Run the paper configuration for all methods and scenes:

```bash
uv run abcd-reproduce \
  --dataset-root datasets \
  --output results/siggraph_2026

# --dataset-root datasets: directory containing garden and kitchen scenes.
# --output results/siggraph_2026: method runs and comparison artifacts.
```

The command reuses completed models and evaluations unless `--force` is given.
It produces `results.json`, `results.csv`, and `comparison.png` in the output
directory. The function defaults are visible in `src/abcd/reproduce.py`; pass
arguments such as `--iterations 10000` to override them.

Equivalent runnable entrypoints live in `scripts/`, for example:

```bash
uv run python scripts/reproduce.py \
  --dataset-root datasets \
  --output results/paper

# --dataset-root datasets: directory containing garden and kitchen scenes.
# --output results/paper: method runs and comparison artifacts.
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

# --output release_checks/memory.json: JSON trace of 1/2/4/8-partition VRAM.
```

The command holds active-shard size fixed across 1, 2, 4, and 8 partitions and
fails when peak allocated VRAM spreads by 1 MiB or more.

## Tests

```bash
uv sync --frozen --group dev
uv run ruff format --check src tests scripts build_native.py
uv run ruff check src tests scripts build_native.py
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
rasterizer and simple-knn code have separate research and evaluation terms.

`vendor/diff-gaussian-rasterization/` is derived from the Gaussian
Splatting software by Inria and MPII, with modifications based on Ashawkey's
rasterizer. Its license at
`vendor/diff-gaussian-rasterization/LICENSE.md` permits research and
evaluation use and prohibits commercial use without prior permission.

`vendor/simple-knn/` is derived from the GraphDECO Gaussian Splatting
implementation by Inria and MPII. Its headers identify the same
Gaussian-Splatting license, so this distribution treats it as research and
evaluation only as well.

Upstream projects: [GraphDECO Gaussian
Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and
[Ashawkey rasterizer](https://github.com/ashawkey/diff-gaussian-rasterization).

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
