import numpy as np
import plyfile

from gs.core.BasePointCloud import BasePointCloud

def read_ply(filename: str) -> BasePointCloud:
    with open(filename) as f:
        data = plyfile.PlyData.read(f)

    # Extract the points and colors
    points = np.vstack([data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]]).T
    colors = np.vstack([data["vertex"]["red"], data["vertex"]["green"], data["vertex"]["blue"]]).T / 255.0

    return BasePointCloud(points, colors)