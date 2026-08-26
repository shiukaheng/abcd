#!/usr/bin/env python
"""Build script for CUDA submodules with compatibility patches."""

import os
import site
import subprocess
import sys
from pathlib import Path

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
submodule_path = Path(
    sys.argv[1] if len(sys.argv) > 1 else "./submodules/diff-gaussian-rasterization"
).resolve()
os.chdir(submodule_path)

result = subprocess.run(
    [sys.executable, "setup.py", "build_ext", "--inplace"],
    env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
)
if result.returncode == 0:
    site_packages = Path(site.getsitepackages()[0])
    path_file = site_packages / "abcd-native.pth"
    existing = set()
    if path_file.exists():
        existing.update(path_file.read_text(encoding="utf-8").splitlines())
    existing.add(str(submodule_path))
    path_file.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")
sys.exit(result.returncode)
