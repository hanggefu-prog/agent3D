"""Maze features must be available before resistance-aware layer assignment."""
import shutil
import subprocess

import pytest

from local_tasks.global_routing_maze_cost.program_adapter import _cost_lambda


def test_maze_lengths_use_initialized_tree_edges_for_sparse_net_ids(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the generated context probe")
    hook = _cost_lambda()
    assert "getNetLength" not in hook
    snapshot = hook.split("  auto open3dApplyMazeStep =", 1)[0]
    source = r"""
#include <algorithm>
#include <cassert>
#include <vector>
struct Net { int getNetLength() = delete; };
struct Edge { int len; };
struct Tree { std::vector<Edge> edges; };
bool open3dMazeIsHbtNet(Net*) { return false; }
double open3dMazeDieId() { return 1.0; }
int main() {
  Net a, b, c;
  std::vector<Net*> nets_{&a, &b, &c};
  std::vector<int> net_ids_{0, 2};
  std::vector<Tree> sttrees_{{{{12}, {8}, {0}, {-1}}}, {{{999}}}, {{{10}}}};
""" + snapshot + r"""
  assert(open3d_net_lengths[0] == 20.0);
  assert(open3d_net_lengths[1] == 0.0);
  assert(open3d_net_lengths[2] == 10.0);
  assert(open3d_net_lengths[0] / (open3d_net_lengths[0] + 80.0) == 0.2);
}
"""
    binary = tmp_path / "probe"
    built = subprocess.run(
        [compiler, "-x", "c++", "-std=c++17", "-O2", "-o", str(binary), "-"],
        input=source, text=True, capture_output=True, timeout=60,
    )
    assert built.returncode == 0, built.stderr
    subprocess.run([str(binary)], check=True, timeout=10)
