::: filecontents*
refs.bib \@inproceedingsli2024nerfxl, author = Li, Ruilong and Fidler,
Sanja and Kanazawa, Angjoo and Williams, Francis, title = NeRF-XL:
Scaling NeRFs with Multiple GPUs, year = 2024, isbn = 978-3-031-73194-5,
publisher = Springer-Verlag, address = Berlin, Heidelberg, url =
https://doi.org/10.1007/978-3-031-73195-2_6, doi =
10.1007/978-3-031-73195-2_6, booktitle = Computer Vision -- ECCV 2024:
18th European Conference, Milan, Italy, September 29--October 4, 2024,
Proceedings, Part XLVIII, pages = 92--107, location = Milan, Italy

\@articlewu2025blockgaussian, title=BlockGaussian: Efficient Large-Scale
Scene Novel View Synthesis via Adaptive Block-Based Gaussian Splatting,
author=Wu, Yongchang and Qi, Zipeng and Shi, Zhenwei and Zou, Zhengxia,
journal=arXiv preprint arXiv:2504.09048, year=2025

\@inproceedingsturki2022mega, title=Mega-NeRF: Scalable Construction of
Large-Scale NeRFs for Virtual Fly-Throughs, author=Turki, Haithem and
Ramanan, Deva and Satyanarayanan, Mahadev, booktitle=Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages=12922--12931, year=2022

\@articlekerbl20233d, title=3d gaussian splatting for real-time radiance
field rendering., author=Kerbl, Bernhard and Kopanas, Georgios and
Leimkühler, Thomas and Drettakis, George and others, journal=ACM Trans.
Graph., volume=42, number=4, pages=139--1, year=2023

\@articlemildenhall2021nerf, title=NeRF: Representing Scenes as Neural
Radiance Fields for View Synthesis, author=Mildenhall, Ben and
Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and
Ramamoorthi, Ravi and Ng, Ren, journal=Communications of the ACM,
volume=65, number=1, pages=99--106, year=2021, publisher=ACM New York,
NY, USA

\@articlewindisch2025lod, title=A LOD of Gaussians: Unified Training and
Rendering for Ultra-Large Scale Reconstruction with External Memory,
author=Windisch, Felix and Köhler, Thomas and Radl, Lukas and D'Urso,
Mattia and Steiner, Michael and Schmalstieg, Dieter and Steinberger,
Markus, journal=arXiv preprint arXiv:2507.01110, year=2025

\@articlewright2015coordinate, title=Coordinate Descent Algorithms,
author=Wright, Stephen J, journal=Mathematical Programming, volume=151,
number=1, pages=3--34, year=2015, publisher=Springer
:::

<figure data-latex-placement="h!">
<img src="./header.png" />
</figure>

# Abstract

We present Space-Time Sharding, a generalized out-of-core training
framework for alpha-composited radiance fields, instantiated here for 3D
Gaussian Splatting. Our method reformulates training as block coordinate
descent over spatial partitions: only one block of parameters is active
at a time, while all others are frozen. By exploiting the associativity
of alpha blending, these inactive regions can be pre-rendered and
collapsed into foreground and background RGBA images.

As a result, for fixed partition size, peak VRAM becomes $O(1)$ with
respect to total scene extent, rather than growing with full scene size.
This enables GPUs with limited memory to train scenes that would
otherwise not fit in core. In experiments, our method closely preserves
reconstruction quality of the original 3DGS, with less than 5% PSNR
degradation, while a naive split-and-merge baseline suffers roughly 40%
degradation.

# Introduction

Radiance fields are seeing increasing industry adoption, and their
application is expanding to ever larger scenes. However, training such
scenes quickly reaches the memory limits of modern GPUs, often requiring
multi-GPU distributed setups that are prohibitively expensive for many
users.

A natural first approach is to split a scene into smaller partitions,
train them independently, and merge them afterward. However,
NeRF-XL [@li2024nerfxl] shows that this often introduces severe
artifacts, since images frequently observe geometry spanning multiple
partitions, causing each partition to compensate for missing content
from the others. Works like BlockGaussians [@wu2025blockgaussian] and
Mega-NeRF [@turki2022mega] improve the naive partition-train-merge
pipeline through better partitioning, such as overlaps or auxiliary
primitives near partition boundaries, but still depend on the quality of
the decomposition. A LOD of Gaussians [@windisch2025lod] reduces memory
with a 3DGS-specific LOD tree, but these approaches are not general to
arbitrary radiance field formulations.

To address this gap, we present Space-Time Sharding, a principled
out-of-core training method for alpha-composited radiance fields. Our
formulation is block coordinate descent over spatial
partitions [@wright2015coordinate], but exploits alpha compositing to
pre-integrate all inactive parameters into foreground and background
images. As a result, each update requires GPU memory only for the active
partition and these two images. We demonstrate the method on 3D Gaussian
Splatting [@kerbl20233d].

# Method

Let the full scene parameters be partitioned into spatial blocks,
$\Theta = \{\theta_i\}$ where each $\theta_i$ corresponds to a spatial
partition $P_i$. At each stage of block coordinate descent, one block is
selected as the active parameters $\theta_{\mathrm{active}}$, while all
remaining parameters are held fixed.

As illustrated in the header figure, for a given active convex partition
and camera view, the frozen parameters can be separated geometrically
into those lying entirely in front of the active region and those lying
entirely behind it. We denote these sets by $\Theta_{\mathrm{fg}}$ and
$\Theta_{\mathrm{bg}}$, respectively. By associativity of alpha
compositing, their rendered contributions can be pre-accumulated into
two flattened RGBA images: a foreground image $F$ and background image
$B$. Optimizing $\theta_{\mathrm{active}}$ while compositing with $F$
and $B$ is equivalent to rendering with all frozen parameters explicitly
present.

We first initialize scene primitives and divide them into convex spatial
partitions (we use a regular 3D grid), yielding parameter blocks
$\{\theta_i\}$. Next, for visible camera--partition pairs, we render and
store compressed RGBA images on disk. To optimize a chosen active block
$\theta_{\mathrm{active}}$, we gather cameras that observe its
partition. For each such camera, cached renders from frozen partitions
are composited into $F$ and $B$ according to depth ordering.

We then perform $T$ gradient steps on $\theta_{\mathrm{active}}$. At
each step, a training camera observing the active partition is sampled,
only the active partition is rendered, the result is composited with $F$
and $B$, reconstruction loss is evaluated, and gradients are
backpropagated only through $\theta_{\mathrm{active}}$. After completing
the $T$ updates, cached renders associated with that partition are
refreshed, and optimization proceeds to the next block.

This produces a flexible memory hierarchy. Disk storage contains cached
visible camera--partition renders. System memory may hold compressed
cached images for faster reuse. GPU memory contains only
$\theta_{\mathrm{active}}$ together with the required
foreground/background images, rather than the full parameter set
$\Theta$, and can be reduced further by streaming a single image pair at
a time.

# Results

We evaluate our method on two scenes from the original NeRF
dataset [@mildenhall2021nerf], *kitchen* and *garden*, each containing
approximately 200 training cameras. For both scenes, we partition the
space using a regular grid with side length 5 units. We compare three
training setups: original 3D Gaussian Splatting (3DGS), naive
split-and-merge training, and Space-Time Sharding.

<figure id="fig:qualitative_comparison" data-latex-placement="t">
<figure>
<img src="./vanilla.png" />
<figcaption>Vanilla 3DGS</figcaption>
</figure>
<figure>
<img src="./naive.png" />
<figcaption>Naive split-and-merge</figcaption>
</figure>
<figure>
<img src="./space-time.png" />
<figcaption>Space-Time Sharding</figcaption>
</figure>
<figcaption>Qualitative comparison on a held-out view. Naive
split-and-merge introduces severe boundary and consistency artifacts,
while Space-Time Sharding closely matches the visual quality of vanilla
3DGS.</figcaption>
</figure>

::: {#tab:results}
+-------------+------------+----------------+----------------+----------------+--------------+--------------+
| **Dataset** | **Method** | **VRAM (GB)    | **RAM (GB)     | **Time (s)     | **PSNR       | **SSIM       |
|             |            | $\downarrow$** | $\downarrow$** | $\downarrow$** | $\uparrow$** | $\uparrow$** |
+:============+:===========+:===============+:===============+:===============+:=============+:=============+
| Garden      | Vanilla    | 0.88           | 7.88           | 99.6           | 24.01        | 0.596        |
|             +------------+----------------+----------------+----------------+--------------+--------------+
|             | Naive      | 0.69           | 8.98           | 402.7          | 16.56        | 0.356        |
|             +------------+----------------+----------------+----------------+--------------+--------------+
|             | Ours       | 0.80           | 25.02          | 1081.3         | 23.65        | 0.575        |
+-------------+------------+----------------+----------------+----------------+--------------+--------------+
| Kitchen     | Vanilla    | 0.97           | 3.73           | 67.4           | 27.03        | 0.861        |
|             +------------+----------------+----------------+----------------+--------------+--------------+
|             | Naive      | 0.75           | 5.60           | 441.3          | 15.64        | 0.504        |
|             +------------+----------------+----------------+----------------+--------------+--------------+
|             | Ours       | 0.77           | 17.73          | 1041.0         | 25.80        | 0.839        |
+-------------+------------+----------------+----------------+----------------+--------------+--------------+

: Comparison across methods on *garden* and *kitchen* scenes
:::

We treat original 3DGS as the quality reference, and use naive grid
partitioning and Space-Time Sharding as an ablation of our technique.
Averaged across both scenes, naive split-and-merge reduces PSNR from
25.52 to 16.10, a 36.9% degradation relative to the reference.
Space-Time Sharding improves this substantially, reaching an average
PSNR of 24.73, which is only a 3.1% drop from vanilla 3DGS. The same
trend appears in SSIM: naive partitioning reduces average SSIM from
0.728 to 0.430, a 41.0% drop, whereas Space-Time Sharding reaches 0.707,
only 3.0% below the reference.

The more significant result is memory scaling. In conventional training,
VRAM grows with total scene size because all parameters must remain
resident during optimization. Under Space-Time Sharding, VRAM depends
only on $\theta_{\mathrm{active}}$ together with one
foreground/background image pair for the sampled view, making memory
effectively independent of total scene extent for fixed partition size.
Averaged across scenes, peak VRAM is 0.785 GB for Space-Time Sharding,
compared with 0.720 GB for naive partitioning and 0.925 GB for vanilla
3DGS. Thus, relative to naive partitioned training, our method increases
VRAM by only 0.065 GB on average, or 9.0%, while still remaining 15.1%
below vanilla 3DGS.

System RAM usage is higher: averaged across scenes, Space-Time Sharding
uses 21.38 GB, compared with 7.29 GB for naive partitioning and 5.81 GB
for vanilla 3DGS. Relative to naive partitioning, this is an increase of
14.09 GB, or 193.2%. This additional memory stores cached rendered
images required by cameras that observe the active partition after
frustum filtering. These images are accessed only when switching active
partitions; many optimization steps are then performed on the same
shard, so transfer cost is strongly amortized. If desired, the cache can
therefore be stored on disk and streamed with limited impact on total
training time.

This memory tradeoff also incurs additional runtime. Averaged across
scenes, training time increases from 83.5 s for vanilla 3DGS and 422.0 s
for naive partitioning to 1061.2 s for Space-Time Sharding. Relative to
naive partitioning, this is an increase of 639.2 s, or 151.5%.

Overall, our method shifts the limiting resource for large-scene
training from scarce GPU memory to cheaper storage tiers, while
preserving nearly the reconstruction quality of unmodified 3DGS.

# Limitations and Future Work

Our current experiments are limited by time and engineering effort. In
particular, we did not fully tune densification and training
hyperparameters for 3D Gaussian Splatting. As a result, VRAM usage in
these experiments is still significantly influenced by image and
pipeline overheads rather than Gaussian parameters alone, which likely
understates the memory advantage of our method relative to vanilla 3DGS.

A practical limitation is that training a partition requires loading
cached render pairs for cameras that observe that partition. With only
frustum culling, this quantity can still grow with scene scale, although
more slowly than storing the full model. In practice, distant partitions
eventually occupy negligible image area and can be ignored, while many
others become occluded. Incorporating visibility thresholds and dynamic
occlusion culling should further reduce memory and storage costs.

Theoretical convergence behavior also remains open. Our method follows
block coordinate descent, which is often effective for smooth
objectives [@wright2015coordinate], but it would be valuable to better
understand its behavior for Gaussian Splatting specifically, especially
when variables are grouped spatially into grid cells.

Our current Gaussian assignment is also approximate: Gaussians are
assigned to partitions using their means, even though their covariance
may cross partition boundaries. Since strict partition membership
determines correctness of the foreground/background decomposition, more
careful assignment rules or overlapping boundary regions could further
improve robustness.

More broadly, partitioned processing of radiance fields appears
promising beyond out-of-core training. While we focus on single-GPU
block coordinate descent, future work could distribute partitions across
multiple compute nodes and exchange only rendered partition images
intermittently. This would resemble asynchronous stochastic optimization
with delayed updates, and may enable decentralized or bandwidth-limited
large-scale Gaussian Splatting training.
