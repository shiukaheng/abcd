#!/usr/bin/env python
"""Build script for CUDA submodules with compatibility patches."""

import os
import sys
import subprocess

# Patch pkg_resources for torch <= 2.0 with newer setuptools
try:
    import pkg_resources

    if not hasattr(pkg_resources, "packaging"):
        import packaging

        pkg_resources.packaging = packaging
        sys.modules["pkg_resources.packaging"] = packaging
except ImportError:
    pass

# Now build the submodule
submodule_path = (
    sys.argv[1] if len(sys.argv) > 1 else "./submodules/diff-gaussian-rasterization"
)
os.chdir(submodule_path)

result = subprocess.run(
    [sys.executable, "setup.py", "build_ext", "--inplace"],
    env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
)
sys.exit(result.returncode)
