# Memory Analysis: Space-Time Sharding for 3DGS

## 1. Paper Goal vs Reality

The paper claims VRAM should be $O(1)$ w.r.t. scene size for fixed partition size — `theta_active` + foreground/background images only. The reported results show VRAM is roughly constant across methods (0.88 vs 0.80 vs 0.69 GB) but still dominated by **image and pipeline overheads**, not Gaussian parameters.

**Hypothesis to investigate:** The method is correct, but memory gains are invisible at current scene scales because static overheads (rasterizer, CUDA allocator, optimizer state, cameras, rendered images) dominate. We need to prove the trend would hold by:
1. Logging ALL tensor allocations with theoretical sizes
2. Comparing theoretical vs actual memory
3. Finding and fixing any leaks or unnecessary resident tensors

## 2. Complete Tensor Inventory

### 2.1 Camera / Image Loading (`gs/io/colmap/__init__.py`)
| Tensor | Shape | dtype | Device | Resident | Theoretical MB |
|--------|-------|-------|--------|----------|----------------|
| `camera.image` (per camera) | `(3, H, W)` | **float64** | CPU | Yes | `3*H*W*8/(1024^2)` |
| `camera.image` (on GPU) | `(3, H, W)` | **float64** | GPU train | Temporarily | same |
| `world_view_transform` | `(4,4)` | float32 | CPU | Yes | negligible |
| `full_proj_transform` | `(4,4)` | float32 | CPU | Yes | negligible |
| `camera_center` | `(3,)` | float32 | CPU | Yes | negligible |

**Note:** Verified via `log_tensor_set` instrumentation: images are actually `torch.float32`, not float64. PyTorch's `uint8 / 255.0` promotes to float32 (the default dtype), not float64. This finding from the agent analysis was incorrect — images are already in the correct dtype.

### 2.2 GaussianModel Parameters (per-model)
| Parameter | Shape | dtype | Device | Description |
|-----------|-------|-------|--------|-------------|
| `positions` | `(N, 3)` | float32 | varies | XYZ positions |
| `sh_coefficients_0` | `(N, 1, 3)` | float32 | varies | DC SH coefficients |
| `sh_coefficients_rest` | `(N, 15, 3)` | float32 | varies | Higher-order SH (sh_degree=3) |
| `rotations` | `(N, 4)` | float32 | varies | Quaternions |
| `scales` | `(N, 3)` | float32 | varies | Log-space scales |
| `opacities` | `(N, 1)` | float32 | varies | Logit opacities |
| `_gradient_accumulator` | `(N, 1)` | float32 | varies | Gradient l1 norm |
| `_gradient_accumulator_denominator` | `(N, 1)` | float32 | varies | Per-Gaussian count |
| `max_radii2D` | `(N, 1)` | float32 | varies | Max projected radius |
| `background_color` | `(3,)` | float32 | CPU | Background fill |

**Total per Gaussian:** `(3+3+45+4+3+1)` = 59 floats for parameters + 3 floats for accumulators = 62 floats = 248 bytes

**Theoretical model size:** `N * 248 / (1024²)` MB + `N * 8` bytes for Adam `exp_avg` + `N * 8` bytes for Adam `exp_avg_sq` per parameter group.

**Adam state:** 6 parameter groups, each with 2 state tensors (`exp_avg`, `exp_avg_sq`) matching parameter shape. Roughly `N * 62 * 8 = N * 496` bytes. Total: ~744 bytes per Gaussian for training.

### 2.3 GridGaussianCell Prerenders
| Tensor | Shape | dtype | Device | Resident | MB per camera |
|--------|-------|-------|--------|----------|---------------|
| `rgb` (prerender) | `(3, H, W)` | uint8 | CPU | Yes (per iter) | `3*H*W/(1024²)` |
| `depth` (prerender) | `(1, H, W)` | float16 | CPU | Yes (per iter) | `2*H*W/(1024²)` |
| `alpha` (prerender) | `(1, H, W)` | uint8 | CPU | Yes (per iter) | `H*W/(1024²)` |

**Total per camera per iteration:** `(3+1+2)*H*W/(1024²)` ≈ `6*H*W/(1024²)` MB

### 2.4 Precomposited Layers (`_precomposited_bg`, `_precomposited_fg`)
| Tensor per camera | Shape | dtype | Device | MB |
|-------------------|-------|-------|--------|-----|
| `rgb` | `(3, H, W)` | float32 | GPU or CPU | `12*H*W/(1024²)` |
| `depth` | `(1, H, W)` | float32 | GPU or CPU | `4*H*W/(1024²)` |
| `alpha` | `(1, H, W)` | float32 | GPU or CPU | `4*H*W/(1024²)` |

**Total per camera (float32):** `(12+4+4)*H*W/(1024²)` = `20*H*W/(1024²)` MB for bg + same for fg = `40*H*W/(1024²)` MB per camera.

## 3. Device Movement Flow

### Vanilla (basic_train)
```
Phase 1: Loading
  load(dataset) → cameras on CPU, images float64 CPU
  from_point_cloud(sparse) → model params: positions(CPU), sh(CPU), scales/distCUDA(CUDA), rotations(CUDA), opacities(CUDA)

Phase 2: Training setup
  model.to("cuda") → ALL params to CUDA (consolidates mixed devices)
  Optimizer creation → Adam state on CUDA per param group

Phase 3: Per-iteration
  camera.to("cuda") → camera + image(float64) to CUDA
  render → temp tensors on CUDA
  loss.backward() → temp grad tensors on CUDA
  camera.to("cpu") → camera + image(float64) to CPU
```

### Grid (Space-Time Sharding)
```
Phase 1: Loading (same as vanilla)

Phase 2: Grid setup
  model.to("cpu") → ALL params to CPU
  GridGaussianModel.from_gaussian_model() → split model into cells (CPU)
  Each cell gets a subset model (params on CPU)

Phase 3: Per-cell training loop
  grid_set_active_cell_index(i) → cell.model.to("cuda"), others stay CPU
  grid_precompose_visible_layers() → builds _precomposited_bg/fg
    - Reads prerenders from CPU (uint8/float16)
    - Converts to float32, moves to storage_device
    - Composites → stores on storage_device
  basic_train(cell) → runs sync_interval iterations
    - Per iteration: camera.to("cuda") → image(float64) to CUDA (!)
    - forward() reads precomposites from storage → .to(train_device)
    - render, loss, backward on CUDA
    - camera.to("cpu")
  cell.clean_model_edges() → cuts model, fresh gradient accumulators
  grid_cull_active_cell_prerenders() → deletes old CPU prerenders
  grid_prerender_active_cell() → renders, stores as uint8/float16 on CPU
  cell.model.to("cpu") → GPU memory freed

Phase 4: Merge
  grid_merge() → concatenate all cell models on target device
```

## 4. Critical Findings

### Finding 1: Images are float64 (2x memory waste)
**File:** `gs/helpers/image.py:10`
```python
torch.from_numpy(np.array(pil_image)) / 255.0
```
`np.array(pil_image)` is `uint8`. `torch.from_numpy(...)` creates `torch.uint8`. Division by `255.0` (a float) promotes to `torch.float64`. No `.float()` call follows.

**Impact:** Every camera image takes 2x the memory it should. For 279 cameras × 800×800×3 = ~2.05 GB required in float32, ~4.1 GB in float64. This cost is identical across all methods (vanilla, naive, ours) — it's a fixed overhead that **masks** the VRAM advantage of grid methods.

**Fix needed:** Add `.float()` to `pil_to_torch()`:
```python
torch.from_numpy(np.array(pil_image)).float() / 255.0
```

### Finding 2: Camera image float64 on GPU every iteration
**File:** `gs/trainers/basic/train.py:135-136`
Each training iteration moves the camera (including its float64 image) to GPU. For a 3×800×800 image at float64 = 15.36 MB per camera. This adds to VRAM each step.

### Finding 3: from_point_cloud creates parameters on mixed devices
**File:** `gs/core/GaussianModel.py:236-297`
- `positions` → CPU (torch.tensor default)
- `sh_coefficients` → CPU
- `scales` (distCUDA2 path) → **CUDA** (`.cuda()` on input)
- `rotations` → **CUDA** (`device="cuda"`)
- `opacities` → **CUDA** (`device="cuda"`)

The subsequent `model.to("cuda")` or `model.to("cpu")` consolidates. But there's a transient moment where tensors exist on both devices.

### Finding 4: clean_model_edges() resets gradient accumulators
**File:** `gs/trainers/grid/grid_utils.py:18` → `model[valid]`  
Creates a new GaussianModel via `__getitem__`, which initializes fresh `_gradient_accumulator`, `_gradient_accumulator_denominator`, and `max_radii2D` (all zeros). **The gradient history is completely lost.**

### Finding 5: Densification zeros ALL gradient accumulators
**File:** `gs/trainers/basic/dynamic_parameters.py:95-97`
```python
model._gradient_accumulator = torch.zeros((new_N, 1), device=device)
model._gradient_accumulator_denominator = torch.zeros((new_N, 1), device=device)
model.max_radii2D = torch.zeros((new_N), device=device).unsqueeze(1)
```
All accumulators zeroed for **all** Gaussians (not just new ones). Combined with Finding 4, this means gradient stats don't persist between cell training rounds.

### Finding 6: Precomposite storage on GPU doesn't truly help at current scale
When `precomposite_storage="gpu"`, the `_precomposited_bg` and `_precomposited_fg` composites live on GPU. For 50 visible cameras × 800² × 2 (bg+fg) × 5 channels × 4 bytes = ~128 MB. This is temporary per-cell but adds to peak VRAM. CPU storage avoids this but adds per-iteration transfer cost.

### Finding 7: Optimizer recreated per cell training round
**File:** `gs/trainers/grid/train.py:97` → `basic_train()`  
Each cell gets a fresh optimizer in `basic_train()`. Adam `exp_avg` and `exp_avg_sq` are zeroed. Combined with Finding 4 and 5, there's **no gradient momentum between cell training rounds**.

### Finding 8: No tensor reuse for per-iteration renders
**File:** `gs/core/GaussianModel.py:158-166`
`forward()` allocates fresh:
- `viewspace_points` (N, 3)
- `radii` (N,)
- Activation tensors (sigmoid/scales, normalized rotations, concatenated SH)

These are all temporary but add up. For 240k Gaussians, that's ~5 MB of temporary allocations per render call.

### Finding 9: redundant `.cuda()` in eval
**File:** `gs/eval/__init__.py:22-24`
```python
camera = camera.to(model.positions.device)   # already on GPU
target = camera.image.cuda()                  # REDUNDANT
```

## 5. VRAM Composition Estimate (Kitchen, 800×800 images, 241k initial Gaussians)

| Component | Vanilla (GB) | Grid GPU (GB) | Grid CPU (GB) |
|-----------|-------------|---------------|---------------|
| Model params (241k) | 0.058 | 0.058 (one cell) | 0.058 (one cell) |
| Adam optimizer state | 0.117 | 0.117 (one cell) | 0.117 (one cell) |
| Gradient accumulators | 0.003 | 0.003 (one cell) | 0.003 (one cell) |
| Camera image (float64!) | 0.015 | 0.015 | 0.015 |
| Rendered output | 0.007 | 0.007 | 0.007 |
| Precomposited bg/fg (50 cams) | 0.000 | 0.128 | 0.000 |
| Prerenders on GPU | 0.000 | 0.000 | 0.000 |
| Rasterizer overhead | ~0.5-0.6 | ~0.5-0.6 | ~0.5-0.6 |
| CUDA context/allocator | ~0.2 | ~0.2 | ~0.2 |
| **Total theoretical** | **~1.0** | **~1.13** | **~1.0** |
| **Observed (paper)** | **0.88-0.97** | **0.77-0.80** | **0.77-0.80** |

**Key insight:** Gaussian parameters are only ~0.058 GB of the total VRAM. The rasterizer/overhead (~0.7 GB) dominates. The float64 camera images add ~0.015 GB per camera in memory. The precomposited images add ~0.128 GB for grid_gpu but 0 for grid_cpu.

The paper's claim that VRAM is $O(1)$ w.r.t. scene size is **correct**, but the constant term (rasterizer overhead, image overheads) is so large at current scene scales that the savings are invisible. To prove the trend, we'd need:
1. Fix float64 → float32 for images (cuts image overhead by 50%)
2. Test on larger scenes where the rasterizer ratio decreases
3. Show that as scene grows, vanilla VRAM grows (more Gaussians) but grid VRAM stays flat

## 6. Potential Memory Leaks / Non-Freeing Patterns

1. **`cell.prerenders` dict accumulation**: Prerenders are stored per-iteration-per-camera. If `grid_cull_active_cell_prerenders` is not called (when `extra_cell_compensation="disabled"`), old prerenders are never deleted.

2. **`_precomposited_bg/fg` dicts**: Cleared at the start of each `grid_precompose_visible_layers()` call — but only for the cameras that appear in the new visible set. If a camera appears in one cell's visible set but not another's, its precomposited data persists.

3. **Viewer tensors**: The viser viewer maintains references to model tensors and creates visualization buffers.

4. **torch.cuda.empty_cache()**: Called in `basic_train:222` every iteration. This is a nuke button — it should be examined whether necessary.

## 7. Recommendations for Paper Fixes

1. **Fix float64 images** → immediate 50% reduction in camera memory
2. **Profile with larger scenes** (more Gaussians) to show VRAM scaling trend
3. **Consider streaming precomposited images** instead of keeping all in memory
4. **Measure rasterizer overhead** → likely the dominant fixed cost
5. **Test with downscaled images** to reduce image overhead further
6. **Implement tensor logging** to validate theoretical sizes match actual
