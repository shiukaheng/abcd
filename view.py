import argparse

from gs.core.GaussianModel import GaussianModel
from gs.visualization.Viewer import Viewer

def main():
    # Only one argument: file path
    parser = argparse.ArgumentParser(description='View a splatting model.')
    parser.add_argument('file', type=str, help='Path to the model file.')
    args = parser.parse_args()

    # Load the model
    model = GaussianModel.from_ply(args.file)
    model.cuda()

    # Create a viewer
    viewer = Viewer(auto_start=False)
    viewer.set_model(model)
    viewer.start(False)

if __name__ == '__main__':
    main()