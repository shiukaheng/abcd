from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "ooc_octree",
        ["src/main.cpp"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="ooc_octree",
    ext_modules=ext_modules,
    # extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
)