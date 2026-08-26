import numpy as np

from abcd.io.colmap.sparse_parsing import read_points3D_text


def test_read_points3d_text_returns_ids_and_values(tmp_path):
    path = tmp_path / "points3D.txt"
    path.write_text(
        "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
        "42 1.0 2.0 3.0 10 20 30 0.25 1 2\n"
        "\n"
        "99 -1.0 0.0 4.0 255 0 7 1.5\n"
    )

    xyz, rgb, errors, point_ids = read_points3D_text(str(path))

    np.testing.assert_allclose(xyz, [[1, 2, 3], [-1, 0, 4]])
    np.testing.assert_array_equal(rgb, [[10, 20, 30], [255, 0, 7]])
    np.testing.assert_allclose(errors, [[0.25], [1.5]])
    np.testing.assert_array_equal(point_ids, [[42], [99]])
