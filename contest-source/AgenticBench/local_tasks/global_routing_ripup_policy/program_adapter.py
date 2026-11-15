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


SOURCE = "OpenROAD-3D/openroad/src/grt/src/fastroute/src/RipUp.cpp"
ENTRYPOINT = "score_ripup_edge"
CONTEXT_FIELDS = {
    "baseline_decision",
    "criticality",
    "detour_ratio",
    "die_id",
    "die_overflow",
    "edge_overflow",
    "has_ndr",
    "history_pressure",
    "is_clock",
    "is_hbt",
    "max_utilization",
    "ripup_threshold",
    "route_length",
}


def _runtime_support(body: str) -> str:
    policy = policy_source(
        body,
        struct_name="Open3DRipupContext",
        function_name="open3dScoreRipupEdge",
        context_fields=CONTEXT_FIELDS,
    )
    return policy.rstrip()[:-len("}  // namespace")] + r'''

std::unordered_set<std::string> open3dHbtPorts()
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

bool open3dIsHbtNet(const FrNet* net)
{
  static const std::unordered_set<std::string> ports = open3dHbtPorts();
  for (const odb::dbBTerm* bterm : net->getDbNet()->getBTerms()) {
    if (ports.contains(bterm->getName())) {
      return true;
    }
  }
  return false;
}

double open3dRoutingDieId()
{
  const char* value = std::getenv("OPEN3D_ROUTING_DIE");
  return value != nullptr && std::string(value) == "top" ? 1.0 : 0.0;
}

}  // namespace
'''


def _hook() -> str:
    return r'''

  // Agentic3D source-evolution hook.  The maintained FastRoute decision has
  // already been computed.  Returning baseline_decision therefore reproduces
  // the original path exactly; the evolved score can also admit a clean but
  // detoured edge or retain an overloaded edge when the global trade-off says
  // that doing so is preferable.
  double open3d_overflow_sum = 0.0;
  double open3d_history_sum = 0.0;
  double open3d_capacity_sum = 0.0;
  double open3d_max_utilization = 0.0;
  for (int open3d_i = 0; open3d_i < treeedge->route.routelen; open3d_i++) {
    const bool open3d_vertical = grids[open3d_i].x == grids[open3d_i + 1].x;
    const int open3d_x = open3d_vertical
                           ? grids[open3d_i].x
                           : std::min(grids[open3d_i].x, grids[open3d_i + 1].x);
    const int open3d_y = open3d_vertical
                           ? std::min(grids[open3d_i].y, grids[open3d_i + 1].y)
                           : grids[open3d_i].y;
    const double open3d_usage = std::max(
        0.0,
        static_cast<double>(open3d_vertical
                                ? graph2d_.getUsageRedV(open3d_x, open3d_y)
                                : graph2d_.getUsageRedH(open3d_x, open3d_y)));
    const double open3d_history = std::max(
        0.0,
        static_cast<double>(open3d_vertical
                                ? graph2d_.getLastUsageV(open3d_x, open3d_y)
                                : graph2d_.getLastUsageH(open3d_x, open3d_y)));
    const double open3d_capacity = std::max(1, open3d_vertical ? v_capacity_
                                                               : h_capacity_);
    open3d_overflow_sum += std::max(0.0, open3d_usage - open3d_capacity);
    open3d_history_sum += open3d_history;
    open3d_capacity_sum += open3d_capacity;
    open3d_max_utilization
        = std::max(open3d_max_utilization,
                   open3d_usage / (open3d_usage + open3d_capacity));
  }
  const double open3d_steps
      = std::max(1, treeedge->route.routelen);
  const double open3d_manhattan
      = static_cast<double>(std::abs(x2 - x1) + std::abs(y2 - y1));
  const double open3d_route_length
      = static_cast<double>(std::max(0, treeedge->route.routelen));
  const float open3d_slack = net->getSlack();
  const bool open3d_sentinel_slack
      = open3d_slack == std::numeric_limits<float>::max()
        || open3d_slack == std::ceil(std::numeric_limits<float>::lowest());
  const double open3d_criticality
      = open3d_sentinel_slack || !std::isfinite(open3d_slack)
            ? 0.0
            : 1.0 / (1.0 + std::max(0.0, static_cast<double>(open3d_slack)));
  Open3DRipupContext open3d_ctx{
      .baseline_decision = needRipup ? 1.0 : 0.0,
      .criticality = open3d_criticality,
      .detour_ratio = open3d_route_length > 0.0
                          ? std::max(0.0, open3d_route_length - open3d_manhattan)
                                / open3d_route_length
                          : 0.0,
      .die_id = open3dRoutingDieId(),
      .die_overflow = static_cast<double>(std::max(0, total_overflow_))
                      / (static_cast<double>(std::max(0, total_overflow_))
                         + std::max<size_t>(1, net_ids_.size())),
      .edge_overflow = open3d_overflow_sum
                       / (open3d_overflow_sum + open3d_steps),
      .has_ndr = net->getDbNet()->getNonDefaultRule() != nullptr ? 1.0 : 0.0,
      .history_pressure = open3d_history_sum
                          / (open3d_history_sum + open3d_capacity_sum),
      .is_clock = net->isClock() ? 1.0 : 0.0,
      .is_hbt = open3dIsHbtNet(net) ? 1.0 : 0.0,
      .max_utilization = open3d_max_utilization,
      .ripup_threshold = 0.5
                         + std::atan(static_cast<double>(ripup_threshold))
                               / 3.14159265358979323846,
      .route_length = open3d_route_length
                      / (open3d_route_length
                         + static_cast<double>(std::max(1, x_grid_ + y_grid_))),
  };
  const double open3d_raw_score = open3dScoreRipupEdge(open3d_ctx);
  const double open3d_score
      = std::isfinite(open3d_raw_score) ? open3d_raw_score
                                       : open3d_ctx.baseline_decision;
  const bool open3d_learned_decision = open3d_score >= 0.5;
  static std::atomic<size_t> open3d_call_counter{0};
  static std::atomic<size_t> open3d_changed_counter{0};
  static std::atomic<size_t> open3d_selected_counter{0};
  const bool open3d_did_change = open3d_learned_decision != needRipup;
  const size_t open3d_calls = open3d_call_counter.fetch_add(1) + 1;
  const size_t open3d_changed
      = open3d_changed_counter.fetch_add(open3d_did_change)
        + open3d_did_change;
  const size_t open3d_selected
      = open3d_selected_counter.fetch_add(open3d_learned_decision)
        + open3d_learned_decision;
  needRipup = open3d_learned_decision;
  if (open3d_calls == 1 || (open3d_did_change && open3d_changed <= 5)
      || open3d_calls % 50000 == 0) {
    logger_->report(
        "OPEN3D_RIPUP_POLICY die={} calls={} changed={} changed_ratio={} "
        "selected={} selected_ratio={} score={}",
        open3d_ctx.die_id > 0.5 ? "top" : "bottom",
        open3d_calls,
        open3d_changed,
        static_cast<double>(open3d_changed) / static_cast<double>(open3d_calls),
        open3d_selected,
        static_cast<double>(open3d_selected) / static_cast<double>(open3d_calls),
        open3d_score);
  }
'''


def _patch_source(before: str, policy: str) -> str:
    after = replace_once(
        before,
        "#include <cstdint>\n",
        "#include <cstdint>\n#include <atomic>\n#include <cstdlib>\n#include <fstream>\n",
        "RipUp standard includes",
    )
    after = replace_once(
        after,
        "#include <limits>\n",
        "#include <limits>\n#include <string>\n#include <unordered_set>\n",
        "RipUp policy includes",
    )
    after = replace_once(
        after,
        '#include "FastRoute.h"\n',
        '#include "FastRoute.h"\n#include "odb/db.h"\n',
        "RipUp database include",
    )
    after = replace_once(
        after,
        "using utl::GRT;\n",
        "using utl::GRT;\n\n" + policy.rstrip() + "\n",
        "RipUp policy insertion",
    )
    anchor = "  if (needRipup) {\n    const int8_t edgeCost = net->getEdgeCost();\n"
    return replace_once(
        after,
        anchor,
        _hook() + "\n" + anchor,
        "newRipupCheck decision hook",
    )


def materialize(program_text: str, open3dbench_root: Path) -> dict:
    body = translate_scalar_program(
        program_text,
        entrypoint=ENTRYPOINT,
        context_fields=CONTEXT_FIELDS,
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
            "Bounded Python rip-up policy transpiled into FastRoute's edge "
            "eligibility decision with die/HBT context and a GRT-only evaluator."
        ),
    }
