# June 10 — Memory Profiling & Logging Infrastructure

## Summary

Built a complete memory profiling pipeline for the Space-Time Sharding 3DGS project:
unified JSONL logging, tensor lifecycle tracking, and Chrome trace visualization.
Ran first comparative benchmarks (vanilla vs grid_cpu on kitchen) and analyzed
the massive gap between theoretical GPU memory and actual VRAM.

## What was built

### 1. Unified JSONL Logger (`gs/profiling/logger.py`)

Replaced two separate CSV loggers (`MemoryMonitor`, `TrainingContext`) with one
`Logger` context manager that writes a single `.jsonl` file. Log types:

| Type | Fields | Source |
|------|--------|--------|
| `memory_snapshot` | `timestamp_s, ram_mb, vram_mb` | Background subprocess polling RAM/VRAM every 100ms |
| `iteration` | `timestamp_s, iteration` | Called from basic_train each iteration |
| `tensor` (event=`set`) | `key, shape, dtype, device, bytes, role` | All tensor creation/replacement points |
| `tensor` (event=`delete`) | `key, reason` | All tensor freeing/destroying points |

Key design: single `mp.Queue` (maxsize 10000) + one writer subprocess. Main process
never blocks (`put_nowait`). `log_tensor_set()` replaces previous value under same key
automatically for the analysis replay.

### 2. Tensor lifecycle logging (8 files instrumented)

`log_tensor_set(key, tensor, role)` — called at these points:

| Event | File:Line | Key pattern |
|-------|-----------|-------------|
| Camera images loaded | `gs/io/colmap/__init__.py` | `cam_{idx}.image` |
| Model params from point cloud | `gs/core/GaussianModel.py` | `model.positions`, `.scales`, etc. |
| Model device movement | `gs/trainers/basic/train.py` | `model.*` (re-logged on new device) |
| Camera GPU/CPU movement | `gs/trainers/basic/train.py` | `cam_{id}.image` |
| Grid cell params created | `gs/trainers/grid/train.py` | `delete model.*`, `set cell_{idx}.*` |
| Cell activation (GPU↔CPU) | `GridGaussianModel.py` | `cell_{idx}.*` re-logged |
| Precomposite bg/fg created | `GridGaussianModel.py` | `cam_{id}.precomp_bg/fg` |
| Prerenders stored | `GridGaussianModel.py` | `cell_{idx}.prerender.{iter}.cam_{id}` |
| Densify (new param sizes) | `dynamic_parameters.py` | `model.*` with new shapes |
| Prune (reduced sizes) | `dynamic_parameters.py` | `model.*` with new shapes |
| Cells merged | `gs/trainers/grid/train.py` | `delete cell_{idx}.*`, `set merged.*` |

### 3. Chrome Trace Converter (`scripts/jsonl_to_trace.py`)

Reads `benchmark.jsonl`, replays tensor events to maintain live memory dict,
emits Chrome Trace Event format at every memory snapshot:

- **MemoryActual** counter: `vram_mb`, `ram_mb` (from system polling)
- **TheoryGPU** counter: `model_mb`, `cameras_mb`, `cells_mb`, `precomp_mb`, `merged_mb` (computed by replaying tensor events on GPU)
- **TheoryCPU** counter: same categories but on CPU
- **Iteration** instant events: markers for each logged iteration

Drag-and-drop into `https://ui.perfetto.dev` for timeline visualization.

### 4. POC test script (`poc_test.sh`)

Single command to run vanilla + grid_cpu on all datasets:
```bash
./poc_test.sh
```

Currently set to kitchen only (counter commented out). Outputs to `new_eval/{dataset}/{method}/` with both `benchmark.jsonl` and `trace.json`.

### 5. Memory analysis doc (`memory_analysis.md`)

Complete inventory of all tensor operations across the codebase (from 4 parallel
exploration agents), device movement flows, and initial findings.

## Bugs found and fixed

1. **`pynvml` scope error** — `_write_snapshot` and `_handle_queue_item` were
   module-level functions referencing `pynvml` which was imported inside
   `_writer_loop`. Fixed by nesting them inside `_writer_loop`.

2. **`_handle_queue_item` orphaned at module level** — A bad edit put the
   writer loop body outside any function. Fixed by restoring the proper
   nesting structure.

3. **`BasePointCloud` import overwritten** — Lost during an edit to add
   `log_tensor_set` import. Restored.

4. **Counter benchmark precomps on GPU** — Earliest counter run used buggy
   logger code. Re-run confirmed precomps on CPU for `grid_cpu` method.

5. **`poc_test.sh` cd path** — Script was at project root, not in `scripts/`
   so `dirname` twice produced wrong directory. Fixed.

## Benchmark results (kitchen, 5000 iters)

| | Vanilla | Grid CPU |
|---|---|---|
| **Peak VRAM** | 982 MB | 814 MB |
| **Peak Theory GPU** | 59 MB | 62 MB |
| **Peak overhead** | 923 MB | 752 MB |
| **Avg VRAM** | 846 MB | 642 MB |
| **Avg Theory GPU** | 61 MB | 27 MB |
| **VRAM/Theory ratio** | **13.9x** | **24.0x** |

Theory GPU breakdown at peak:

| Category | Vanilla | Grid CPU |
|----------|---------|----------|
| model params | 58.6 MB | 29.5 MB |
| cell params | — | 28.2 MB |
| cameras | — | 4.6 MB |
| **Total theory** | **58.6 MB** | **62.3 MB** |
| **Overhead** | **923 MB** | **752 MB** |

### Key finding

Actual VRAM is 14-24x larger than theoretical GPU memory (Gaussian parameters).
The gap (~750-923 MB) is dominated by fixed overhead: rasterizer internal buffers,
CUDA context, PyTorch allocator cache, and optimizer momentum buffers.

The grid method reduces VRAM by ~200 MB (visible advantage), but the ~750 MB
baseline overhead swamps the savings at current scene scale. The theory predicts
that as scene size grows, vanilla VRAM would grow linearly with Gaussians while
grid VRAM stays constant — but the overhead floor means this advantage is hard
to see at <200k Gaussians.

## Directory cleanup

- Removed old `new_eval/counter/` and `eval/` artifacts
- All eval output now goes to `new_eval/{dataset}/{method}/`

## Files changed (commits: `f1cbc77`, `cd353ab`)

```
UNIFY:     gs/profiling/logger.py        (Logger class replaces MemoryMonitor/TrainingContext)
DELETED:   gs/profiling/memory_monitor.py, training_context.py
MODIFIED:  gs/profiling/__init__.py, run_benchmark.py
ADDED:     gs/profiling/logger.py

TENSOR LOG: gs/core/GaussianModel.py     (from_point_cloud logging)
            gs/io/colmap/__init__.py     (camera image logging)
            gs/trainers/basic/train.py   (model/camera device logging)
            gs/trainers/grid/train.py    (grid setup/merge logging)
            GridGaussianModel.py         (cell/prerender/precomp logging)
            dynamic_parameters.py        (densify/prune logging)
            memory_analysis.md           (tensor inventory)
            scripts/jsonl_to_trace.py    (trace converter)
            poc_test.sh                  (benchmark runner)
```

## Next steps / todo

- [ ] Add optimizer state (`exp_avg`/`exp_avg_sq`) to tensor logging — likely
      explains some of the overhead
- [ ] Run `poc_test.sh` with counter dataset for second data point
- [ ] Plot Theory GPU vs Actual VRAM as function of scene Gaussians for paper
- [ ] Investigate if `torch.cuda.empty_cache()` calls are necessary
- [ ] Test `grid_gpu` method for VRAM comparison
- [ ] Fix `model.*` double-counting in grid traces (densify logs `model.*`
      which overlaps with `cell_*.*`)
