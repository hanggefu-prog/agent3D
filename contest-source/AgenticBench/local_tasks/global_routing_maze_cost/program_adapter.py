from __future__ import annotations

from pathlib import Path

from local_bench.program_safety import read_repo_file
from local_tasks.routing_source_common import (
    grt_only_patches,
    materialized_patch,
    policy_source,
    replace_once,
    translate_scalar_program,
)


SOURCE = "OpenROAD-3D/openroad/src/grt/src/fastroute/src/maze.cpp"
ENTRYPOINT = "score_maze_step"
CONTEXT_FIELDS = {
    "baseline_cost",
    "congestion_pressure",
    "criticality",
    "die_id",
    "die_overflow",
    "edge_overflow",
    "history_pressure",
    "is_clock",
    "is_hbt",
    "is_horizontal",
    "is_turn",
    "reroute_phase",
    "wirelength",
}


def _runtime_support(body: str) -> str:
    policy = policy_source(
        body,
        struct_name="Open3DMazeStepContext",
        function_name="open3dScoreMazeStep",
        context_fields=CONTEXT_FIELDS,
    )
    return policy.rstrip()[:-len("}  // namespace")] + r'''

std::unordered_set<std::string> open3dMazeHbtPorts()
{
  std::unordered_set<std::string> ports;
  if (const char* path = std::getenv("OPEN3D_HBT_PORT_FILE")) {
    std::ifstream stream(path);
    std::string row;
    while (std::getline(stream, row)) {
      const size_t first = row.find('\t');
      if (first == std::string::npos) {
        continue;
      }
      const size_t second = row.find('\t', first + 1);
      ports.insert(row.substr(first + 1, second - first - 1));
    }
  }
  return ports;
}

bool open3dMazeIsHbtNet(const FrNet* net)
{
  static const std::unordered_set<std::string> ports = open3dMazeHbtPorts();
  for (const odb::dbBTerm* bterm : net->getDbNet()->getBTerms()) {
    if (ports.contains(bterm->getName())) {
      return true;
    }
  }
  return false;
}

double open3dMazeDieId()
{
  const char* value = std::getenv("OPEN3D_ROUTING_DIE");
  return value != nullptr && std::string(value) == "top" ? 1.0 : 0.0;
}

}  // namespace
'''


def _cost_lambda() -> str:
    return r'''
  // Agentic3D source-evolution hook.  The candidate returns a multiplier for
  // one actual maze expansion.  A multiplier of 1.0 is bit-for-bit equivalent
  // to the maintained edge cost.  Trusted clamps retain positive finite costs
  // so Dijkstra's assumptions and deterministic fallback remain intact.
  std::vector<bool> open3d_is_hbt(nets_.size(), false);
  // FrNet::net_length_ is populated by the resistance-aware layer-assignment
  // path, which has not run at this 2-D maze boundary. Read the initialized
  // Steiner edges instead, once per maze iteration rather than per expansion.
  std::vector<double> open3d_net_lengths(nets_.size(), 0.0);
  for (const int open3d_net_id : net_ids_) {
    open3d_is_hbt[open3d_net_id]
        = open3dMazeIsHbtNet(nets_[open3d_net_id]);
    for (const auto& open3d_edge : sttrees_[open3d_net_id].edges) {
      open3d_net_lengths[open3d_net_id] += std::max(0, open3d_edge.len);
    }
  }
  const double open3d_die_id = open3dMazeDieId();
  auto open3dApplyMazeStep = [&](const double baseline_cost,
                                 const int x,
                                 const int y,
                                 const bool is_horizontal,
                                 const bool is_turn,
                                 const int net_id) {
    FrNet* open3d_net = nets_[net_id];
    const double open3d_usage = std::max(
        0.0,
        static_cast<double>(is_horizontal ? graph2d_.getUsageRedH(x, y)
                                          : graph2d_.getUsageRedV(x, y)));
    const double open3d_history = std::max(
        0.0,
        static_cast<double>(is_horizontal ? graph2d_.getLastUsageH(x, y)
                                          : graph2d_.getLastUsageV(x, y)));
    const double open3d_capacity
        = std::max(1, is_horizontal ? h_capacity_ : v_capacity_);
    const double open3d_overflow
        = std::max(0.0, open3d_usage - open3d_capacity);
    const float open3d_slack = open3d_net->getSlack();
    const bool open3d_sentinel_slack
        = open3d_slack == std::numeric_limits<float>::max()
          || open3d_slack == std::ceil(std::numeric_limits<float>::lowest());
    const double open3d_criticality
        = open3d_sentinel_slack || !std::isfinite(open3d_slack)
              ? 0.0
              : 1.0 / (1.0 + std::max(0.0, static_cast<double>(open3d_slack)));
    const double open3d_wirelength
        = open3d_net_lengths[net_id];
    Open3DMazeStepContext open3d_ctx{
        .baseline_cost = baseline_cost
                         / (baseline_cost
                            + std::max(
                                1.0,
                                static_cast<double>(cost_params.cost_height))),
        .congestion_pressure = open3d_usage
                               / (open3d_usage + open3d_capacity),
        .criticality = open3d_criticality,
        .die_id = open3d_die_id,
        .die_overflow = static_cast<double>(std::max(0, total_overflow_))
                        / (static_cast<double>(std::max(0, total_overflow_))
                           + std::max<size_t>(1, net_ids_.size())),
        .edge_overflow = open3d_overflow
                         / (open3d_overflow + open3d_capacity),
        .history_pressure = open3d_history
                            / (open3d_history + open3d_capacity),
        .is_clock = open3d_net->isClock() ? 1.0 : 0.0,
        .is_hbt = open3d_is_hbt[net_id] ? 1.0 : 0.0,
        .is_horizontal = is_horizontal ? 1.0 : 0.0,
        .is_turn = is_turn ? 1.0 : 0.0,
        .reroute_phase = static_cast<double>(std::max(0, iter))
                         / static_cast<double>(std::max(1, iter + 8)),
        .wirelength = open3d_wirelength
                      / (open3d_wirelength
                         + static_cast<double>(std::max(1, x_grid_ + y_grid_))),
    };
    const double open3d_raw_multiplier = open3dScoreMazeStep(open3d_ctx);
    const double open3d_multiplier
        = std::isfinite(open3d_raw_multiplier)
              ? std::clamp(open3d_raw_multiplier, 0.05, 20.0)
              : 1.0;
    static std::atomic<size_t> open3d_call_counter{0};
    static std::atomic<size_t> open3d_changed_counter{0};
    const bool open3d_did_change
        = std::abs(open3d_multiplier - 1.0) > 1.0e-12;
    const size_t open3d_calls = open3d_call_counter.fetch_add(1) + 1;
    const size_t open3d_changed
        = open3d_changed_counter.fetch_add(open3d_did_change)
          + open3d_did_change;
    if (open3d_calls == 1 || (open3d_did_change && open3d_changed <= 5)
        || open3d_calls % 100000 == 0) {
      logger_->report(
          "OPEN3D_MAZE_COST die={} calls={} changed={} changed_ratio={} "
          "multiplier_last={}",
          open3d_ctx.die_id > 0.5 ? "top" : "bottom",
          open3d_calls,
          open3d_changed,
          static_cast<double>(open3d_changed)
              / static_cast<double>(open3d_calls),
          open3d_multiplier);
    }
    return baseline_cost * open3d_multiplier;
  };

'''


def _patch_source(before: str, policy: str) -> str:
    after = replace_once(
        before,
        "#include <cstdint>\n",
        "#include <cstdint>\n#include <atomic>\n#include <cstdlib>\n#include <fstream>\n",
        "maze standard includes",
    )
    after = replace_once(
        after,
        "#include <unordered_map>\n",
        "#include <unordered_map>\n#include <unordered_set>\n#include <string>\n",
        "maze policy includes",
    )
    # OpenROAD releases differ in whether maze.cpp directly includes Graph2D.h
    # and exposes Timer here.  Bind both supported layouts to an exact unique
    # anchor; never rely on a fuzzy patch or duplicate an existing db include.
    if '#include "odb/db.h"\n' not in after:
        database_anchor = (
            '#include "Graph2D.h"\n'
            if after.count('#include "Graph2D.h"\n') == 1
            else '#include "FastRoute.h"\n'
        )
        after = replace_once(
            after,
            database_anchor,
            database_anchor + '#include "odb/db.h"\n',
            "maze database include",
        )
    policy_anchor = (
        "using utl::Timer;\n"
        if after.count("using utl::Timer;\n") == 1
        else "using utl::GRT;\n"
    )
    after = replace_once(
        after,
        policy_anchor,
        policy_anchor + "\n" + policy.rstrip() + "\n",
        "maze policy insertion",
    )
    after = replace_once(
        after,
        "  auto relaxAdjacent = [&](const int cur_x,\n",
        _cost_lambda() + "  auto relaxAdjacent = [&](const int cur_x,\n",
        "maze relaxation hook",
    )
    after = replace_once(
        after,
        "    double cost1 = getCost(pos1, is_horizontal, cost_params);\n",
        "    double cost1 = open3dApplyMazeStep(\n"
        "        getCost(pos1, is_horizontal, cost_params),\n"
        "        p1_x, p1_y, is_horizontal, add_via, net_id);\n",
        "primary maze step cost",
    )
    return replace_once(
        after,
        "        double cost2 = getCost(pos2, is_horizontal, cost_params);\n",
        "        double cost2 = open3dApplyMazeStep(\n"
        "            getCost(pos2, is_horizontal, cost_params),\n"
        "            p2_x, p2_y, is_horizontal, add_via, net_id);\n",
        "hyper maze step cost",
    )


def materialize(program_text: str, open3dbench_root: Path) -> dict:
    body = translate_scalar_program(
        program_text,
        entrypoint=ENTRYPOINT,
        context_fields=CONTEXT_FIELDS,
        max_static_loop_iterations=8,
        max_nested_loop_iterations=16,
    )
    source_before = read_repo_file(open3dbench_root, SOURCE)
    source_after = _patch_source(source_before, _runtime_support(body))
    route_after, runner_after = grt_only_patches(open3dbench_root)
    return {
        "schema_version": 1,
        "config_overrides": {},
        "candidate_files": {},
        "source_patch": materialized_patch(
            source_path=SOURCE,
            source_before=source_before,
            source_after=source_after,
            route_after=route_after,
            runner_after=runner_after,
            open3dbench_root=open3dbench_root,
        ),
        "notes": (
            "Bounded Python maze-step multiplier transpiled into FastRoute's "
            "real relaxation cost with die/HBT context and trusted cost clamps."
        ),
    }
