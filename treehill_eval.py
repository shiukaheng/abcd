from gs.core.GaussianModel import GaussianModel
from gs.eval import eval_views
from gs.io.colmap import load

if __name__ == "__main__":
    cameras, sparse = load("./datasets/mip_nerf_360/treehill/")
    model = GaussianModel.from_ply("./samples/treehill_grid_bench.ply").cuda()
    model.assert_validity()
    r = eval_views(cameras, model)
    