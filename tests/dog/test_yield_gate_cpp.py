"""The C++ yield gate in Nav_D20/planning (src/algorithm/yield_gate/) against the Python reference, frame by frame:
hand-made and random scenarios (Nav_D20/planning/tools/yield_gate_parity/). Needs g++ and Eigen 3."""
import os
import shutil
import subprocess
import sys

import pytest

from .conftest import ROOT

EIGEN = os.environ.get("EIGEN_DIR", "/usr/include/eigen3")


def test_cpp_yield_gate_agrees_with_python(tmp_path):
    if shutil.which("g++") is None or not os.path.isdir(EIGEN):
        pytest.skip("needs g++ and Eigen 3")
    run = os.path.join(ROOT, "Nav_D20", "planning", "tools", "yield_gate_parity", "run.sh")
    r = subprocess.run(["bash", run, "--random", "150", "--seed", "3"], capture_output=True, text=True,
                       env=dict(os.environ, YIELD_PARITY_DIR=str(tmp_path), PYTHON=sys.executable), timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert " 0 differ" in r.stdout, r.stdout[-3000:]
