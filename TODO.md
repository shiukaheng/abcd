# Software engineering
- [ ] For non-optimization related variables, refactor to use numpy instead of tensors
# Camera poses
- [ ] DUST3R based SfM replacement (DUST3R, InstantSplat)
- [ ] COLMAP acceleration by using neural networks to sparsify image graph
We deliberately don't want to optimize camera poses jointly with Gaussians, since by assuming constant camera poses, we can use our grid-based optimization to optimize the Gaussians with much lower memory requirements. (As compared to InstantSplat)
# Gaussian initialization
- [ ] DUST3R based initialization (DUST3R, InstantSplat)
# Optimization
- [x] Reparameterizing Gaussian sizes with sigmoid activation instead of exponential to limit size to specific range
    - Large Gaussians faultily attempt to model per-camera variations, which is not the intended use case
    - Large Gaussians are EXTREMELY slow to render
    - Large Gaussians become distorted in the projection process due to first-degree linearization (GS++)
- [ ] Normal computation (GaussianPro)
    - [ ] Post-processing depth map to compute normals
    - [ ] OR: 
- [ ] Depth and normal supervision, accounting for confidence using DUST3R (https://arxiv.org/abs/2003.10432) (GaussianPro, NOT done in InstantSplat)
- [ ] Normal smoothing loss (GaussianPro)
- [ ] Dense / linear layer before outputting spherical harmonic coefficients, with additional info based on camera identity, posed as embedding layer
    - [ ] We can bake in the final spherical harmonics during export by pre-applying the linear layer, resulting in the spherical harmonics only.
- [ ] Alpha regularization: Every pixel should have a non-zero alpha value!
# Memory optimization
- [ ] Reduce relevant cameras per chunk
    - [ ] Calculate "contribution" of camera to chunk, based on compsiting the other chunks in. Discard cameras as they become irrelevant.
- [ ] Sparsify voxel grid dependency by setting hard limit, or by iteratively gainign better understanding of unrelated voxels
    - [ ] Unrelated voxels do not have to be stored in memory, rather on disk
- [ ] Save unused models to disk to save RAM
# Rendering
- [ ] Address close-up issues
    - Uncertainty needs to be properly modelled to avoid high frequency artifacts (MIP-Splatting)
- [ ] Address far-away issues
    - MIP-Splatting replaces 2D dilation filter with 3D smoothing filter. However, we want to skip this because it requires custom rendering pipeline that is hard to adopt to different platforms.