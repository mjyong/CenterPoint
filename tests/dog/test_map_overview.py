"""Nav_D20/planning/src/runnable/preprocess/global_map_manager/map_overview.h: the coarse global map the planner
publishes for display (T59 review: the site map's full grid is 117 MB, the dog's shared memory takes 16 MB chunks).
Compiled on its own with g++ and Eigen; checks the size cap, the pooling and the orientation against grid_map's
index convention (index (0, 0) at the max x, max y corner)."""
import os
import shutil
import subprocess

import pytest

from .conftest import ROOT

HEADER_DIR = os.path.join(ROOT, "Nav_D20", "planning")
EIGEN = os.environ.get("EIGEN_DIR", "/usr/include/eigen3")

PROGRAM = r"""
#include "src/runnable/preprocess/global_map_manager/map_overview.h"
#include <cstdio>
#include <limits>
using planning::preprocess::BuildMapOverview;
int main() {
  // 10 x 7 cells at 0.1 m, max corner (5.0, 3.0): rows along -x, columns along -y; all free
  Eigen::MatrixXf m = Eigen::MatrixXf::Constant(10, 7, 255.0f);
  m(0, 0) = 0.0f;                                    // blocked: the cell at the max x, max y corner
  m(9, 6) = std::numeric_limits<float>::quiet_NaN(); // unknown: the min corner
  auto full = BuildMapOverview(m, 0.1, 5.0, 3.0, 1000);
  std::printf("full %d %d %.3f %.3f %.3f max %d min %d\n", full.width, full.height, full.resolution, full.origin_x,
              full.origin_y, full.data[(full.height - 1) * full.width + full.width - 1], full.data[0]);
  // at most 12 cells: 2 x 2 gives 5 x 4 = 20, too many; pooled 3 x 3 -> 4 x 3 = 12 cells, 0.3 m
  auto o = BuildMapOverview(m, 0.1, 5.0, 3.0, 12);
  int blocked = 0, unknown = 0;
  for (auto v : o.data) { blocked += v == 100; unknown += v == -1; }
  std::printf("coarse %d %d %.3f %.3f %.3f blocked %d at %d unknown %d\n", o.width, o.height, o.resolution, o.origin_x,
              o.origin_y, blocked, (o.height - 1) * o.width + o.width - 1 == 11 && o.data[11] == 100, unknown);
  // the site map's size: 10293 x 11412 at 0.1 m, capped at 4 M cells
  Eigen::MatrixXf big = Eigen::MatrixXf::Constant(10293, 11412, 255.0f);
  auto b = BuildMapOverview(big, 0.1, 0.0, 0.0, 4000000);
  std::printf("site %d %d %.2f %zu\n", b.width, b.height, b.resolution, b.data.size());
  return 0;
}
"""


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    if shutil.which("g++") is None or not os.path.isdir(EIGEN):
        pytest.skip("needs g++ and Eigen 3")
    d = tmp_path_factory.mktemp("overview")
    src, exe = d / "t.cc", d / "t"
    src.write_text(PROGRAM)
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", HEADER_DIR, "-I", EIGEN, str(src), "-o", str(exe)], check=True)
    return subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines()


def test_full_resolution_keeps_the_corners_where_they_are(run):
    # nothing pooled: 10 cells along x, 7 along y, origin at the min corner (5.0 - 1.0, 3.0 - 0.7); the blocked cell
    # (grid_map index (0, 0)) is the last in OccupancyGrid order, the NaN one (index (9, 6)) the first
    assert run[0] == "full 10 7 0.100 4.000 2.300 max 100 min -1"


def test_pooling_caps_the_cells_and_keeps_a_blocked_cell(run):
    # k = 3: ceil(10/3) = 4 along x, ceil(7/3) = 3 along y, origin 5.0 - 1.2, 3.0 - 0.9; the block with the blocked
    # cell is blocked (the last cell); the min-corner block is the NaN cell alone (row 9, column 6): unknown
    assert run[1] == "coarse 4 3 0.300 3.800 2.100 blocked 1 at 1 unknown 1"


def test_the_site_map_fits_the_shared_memory(run):
    width, height, res, cells = run[2].split()[1:]
    assert float(res) == pytest.approx(0.6) and int(cells) <= 4000000     # 3.3 MB, under the 16 MB chunks
    assert (int(width), int(height)) == (1716, 1902)
