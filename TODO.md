- [ ] For non-optimization related variables, refactor to use numpy instead of tensors

# Camera poses
- [ ] DUST3R based SfM replacement (?)
- [ ] COLMAP acceleration by using neural networks to sparsify image graph
# Gaussian initialization
- [ ] DUST3R based initialization
# Optimization
- [ ] Depth and normal supervision, accounting for confidence using DUST3R (https://arxiv.org/abs/2003.10432)
- [ ] Normal smoothing loss
- [ ] Dense / linear layer before outputting spherical harmonic coefficients, with additional info based on camera identity, posed as embedding layer
    - [ ] We can bake in the final spherical harmonics during export by pre-applying the linear layer, resulting in the spherical harmonics only.
# Memory optimization
- [ ] Sparsify voxel grid dependency by setting hard limit, or by iteratively gainign better understanding of unrelated voxels