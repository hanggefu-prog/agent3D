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


SOURCE = "OpenROAD-3D/openroad/src/grt/src/fastroute/src/FastRoute.cpp"
ENTRYPOINT = "control_reroute_iteration"
CONTEXT_FIELDS = {
    "baseline_congestion_scale",
    "baseline_ripup_offset",
    "baseline_search_scale",
    "congestion_trend",
    "cost_height",
    "critical_pressure",
    "die_id",
    "die_overflow",
    "hbt_pressure",
    "history_weight",
    "max_overflow",
    "ndr_pressure",
    "reroute_phase",
    "ripup_threshold",
    "search_window",
    "stagnation",
    "wirelength_pressure",
}


def _runtime_support(body: str) -> str:
    policy = policy_source(
        body,
        struct_name="Open3DIterationContext",
        function_name="open3dControlRerouteIteration",
        context_fields=CONTEXT_FIELDS,
        return_type="Open3DIterationDecision",
        extra_declarations=r'''struct Open3DIterationDecision
{
  double congestion_scale;
  double search_scale;
  double ripup_offset;
};''',
    )
    return policy.rstrip()[:-len("}  // namespace")] + r'''

std::unordered_set<std::string> open3dIterationHbtPorts()
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

bool open3dIterationIsHbtNet(const FrNet* net)
{
  static const std::unordered_set<std::string> ports = open3dIterationHbtPorts();
  for (const odb::dbBTerm* bterm : net->getDbNet()->getBTerms()) {
    if (ports.contains(bterm->getName())) {
      return true;
    }
  }
  return false;
}

double open3dIterationDieId()
{
  const char* value = std::getenv("OPEN3D_ROUTING_DIE");
  return value != nullptr && std::string(value) == "top" ? 1.0 : 0.0;
}

}  // namespace
'''


def _controller_hook() -> str:
    return r'''
      // Agentic3D source-evolution hook. The candidate independently controls
      // congestion cost, search-window width, and rip-up permissiveness.
      // Returning (1.0, 1.0, 0.0) exactly preserves all maintained values;
      // trusted clamps keep the three corrections within safe ranges.
      size_t open3d_hbt_nets = 0;
      size_t open3d_ndr_nets = 0;
      size_t open3d_critical_nets = 0;
      for (const int open3d_net_id : net_ids_) {
        FrNet* open3d_net = nets_[open3d_net_id];
        open3d_hbt_nets += open3dIterationIsHbtNet(open3d_net);
        open3d_ndr_nets
            += open3d_net->getDbNet()->getNonDefaultRule() != nullptr;
        open3d_critical_nets += open3d_net->isCritical();
      }
      const double open3d_net_count
          = static_cast<double>(std::max<size_t>(1, net_ids_.size()));
      const double open3d_grid_capacity
          = static_cast<double>(std::max(1, x_grid_ * y_grid_))
            * static_cast<double>(std::max(1, h_capacity_ + v_capacity_));
      const double open3d_total_overflow
          = static_cast<double>(std::max(0, total_overflow_));
      const double open3d_trend
          = last_total_overflow > 0
                ? std::clamp((static_cast<double>(last_total_overflow)
                              - open3d_total_overflow)
                                 / static_cast<double>(last_total_overflow),
                             -1.0,
                             1.0)
                : 0.0;
      static int open3d_best_overflow = std::numeric_limits<int>::max();
      static int open3d_stagnant_iterations = 0;
      if (total_overflow_ < open3d_best_overflow) {
        open3d_best_overflow = total_overflow_;
        open3d_stagnant_iterations = 0;
      } else {
        open3d_stagnant_iterations++;
      }
      Open3DIterationContext open3d_ctx{
          .baseline_congestion_scale = 1.0,
          .baseline_ripup_offset = 0.0,
          .baseline_search_scale = 1.0,
          .congestion_trend = 0.5 + 0.5 * open3d_trend,
          .cost_height = static_cast<double>(std::max(0, costheight_))
                         / (static_cast<double>(std::max(0, costheight_))
                            + 50.0),
          .critical_pressure = static_cast<double>(open3d_critical_nets)
                               / open3d_net_count,
          .die_id = open3dIterationDieId(),
          .die_overflow = open3d_total_overflow
                          / (open3d_total_overflow + open3d_net_count),
          .hbt_pressure = static_cast<double>(open3d_hbt_nets)
                          / open3d_net_count,
          .history_weight = L != 0 ? 1.0 : 0.0,
          .max_overflow = static_cast<double>(std::max(0, maxOverflow))
                          / (static_cast<double>(std::max(0, maxOverflow))
                             + std::max(1, h_capacity_ + v_capacity_)),
          .ndr_pressure = static_cast<double>(open3d_ndr_nets)
                          / open3d_net_count,
          .reroute_phase = static_cast<double>(std::max(0, i))
                           / static_cast<double>(
                               std::max(1, overflow_iterations_)),
          .ripup_threshold = 0.5
                             + std::atan(
                                   static_cast<double>(ripup_threshold))
                                   / 3.14159265358979323846,
          .search_window = static_cast<double>(std::max(0, enlarge_))
                           / static_cast<double>(
                               std::max(1, std::max(x_grid_, y_grid_))),
          .stagnation = static_cast<double>(
                            std::max(0, open3d_stagnant_iterations))
                        / static_cast<double>(
                            std::max(1, open3d_stagnant_iterations + 5)),
          .wirelength_pressure = static_cast<double>(std::max(0, tUsage))
                                 / (static_cast<double>(std::max(0, tUsage))
                                    + open3d_grid_capacity),
      };
      const Open3DIterationDecision open3d_raw_decision
          = open3dControlRerouteIteration(open3d_ctx);
      const double open3d_cost_scale
          = std::isfinite(open3d_raw_decision.congestion_scale)
                ? std::clamp(
                      open3d_raw_decision.congestion_scale, 0.25, 4.0)
                : 1.0;
      const double open3d_search_scale
          = std::isfinite(open3d_raw_decision.search_scale)
                ? std::clamp(open3d_raw_decision.search_scale, 0.50, 2.50)
                : 1.0;
      const double open3d_ripup_offset
          = std::isfinite(open3d_raw_decision.ripup_offset)
                ? std::clamp(open3d_raw_decision.ripup_offset, -4.0, 4.0)
                : 0.0;
      costheight_ = std::max(
          1.0, static_cast<double>(costheight_) * open3d_cost_scale);
      enlarge_ = std::clamp(
          static_cast<int>(std::lround(enlarge_ * open3d_search_scale)),
          1,
          std::max(1, x_grid_ / 2));
      ripup_threshold = std::clamp(
          ripup_threshold
              + static_cast<int>(std::lround(open3d_ripup_offset)),
          -8,
          8);
      logger_->report(
          "OPEN3D_ITERATION_CONTROL die={} calls={} changed={} "
          "cost_scale={} search_scale={} ripup_offset={} cost_height={} "
          "search_window={} ripup_threshold={}",
          open3d_ctx.die_id > 0.5 ? "top" : "bottom",
          i,
          (std::abs(open3d_cost_scale - 1.0) > 1.0e-12
           || std::abs(open3d_search_scale - 1.0) > 1.0e-12
           || std::abs(open3d_ripup_offset) > 1.0e-12)
              ? 1
              : 0,
          open3d_cost_scale,
          open3d_search_scale,
          open3d_ripup_offset,
          costheight_,
          enlarge_,
          ripup_threshold);
'''


def _patch_source(before: str, policy: str) -> str:
    after = replace_once(
        before,
        "#include <cstdint>\n",
        "#include <cstdint>\n#include <cstdlib>\n",
        "FastRoute policy includes",
    )
    after = replace_once(
        after,
        "using utl::GRT;\n",
        "using utl::GRT;\n\n" + policy.rstrip() + "\n",
        "FastRoute policy insertion",
    )
    snapshot_anchor = (
        "      const int snapshot_batch_count_before = snapshot_batch_count_;\n"
    )
    if after.count(snapshot_anchor) == 1:
        return replace_once(
            after,
            snapshot_anchor,
            snapshot_anchor + _controller_hook() + "\n",
            "overflow-iteration controller hook",
        )
    legacy_anchor = (
        "    if (maxOverflow > 300 && past_cong > 15000) {\n"
        "      L = 0;\n"
        "    }\n\n"
        "    auto cost_params = CostParams(logistic_coef, costheight_, slope);\n"
    )
    return replace_once(
        after,
        legacy_anchor,
        "    if (maxOverflow > 300 && past_cong > 15000) {\n"
        "      L = 0;\n"
        "    }\n\n"
        + _controller_hook()
        + "\n    auto cost_params = CostParams(logistic_coef, costheight_, slope);\n",
        "overflow-iteration controller hook",
    )


def materialize(program_text: str, open3dbench_root: Path) -> dict:
    body = translate_scalar_program(
        program_text,
        entrypoint=ENTRYPOINT,
        context_fields=CONTEXT_FIELDS,
        return_arity=3,
        return_cpp_type="Open3DIterationDecision",
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
            "Bounded Python iteration controller transpiled into FastRoute's "
            "overflow loop and mapped to trusted cost/window/rip-up controls."
        ),
    }
