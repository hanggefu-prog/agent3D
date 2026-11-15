"""The complete maintained local-task inventory; retired definitions are removed.

Membership is not evidence of native reference qualification. Each task still
requires an authenticated reference for its selected physical input.
"""

CORE_TASK_DIRECTORIES = (
    "tp_gnn_post_placement_timing",
    "tp_gnn_binary_loss_timing",
    "tp_gnn_loss_decoder_timing",
    "placement_initialization",
    "placement_preconditioner",
    "placement_parameters_reference_20260923",
    "hbt_assignment_policy",
    "global_routing_ripup_policy",
    "global_routing_maze_cost",
    "global_routing_iteration_controller",
)

# Global Analyzer may model only tasks with a connected native evaluator.
# HBT evaluates the post-placement physical handoff before OPT.
EXECUTABLE_TASK_STAGES = {
    "tp_gnn_post_placement_timing": "partition",
    "tp_gnn_binary_loss_timing": "partition",
    "tp_gnn_loss_decoder_timing": "partition",
    "placement_initialization": "layout",
    "placement_preconditioner": "layout",
    "placement_parameters_reference_20260923": "layout",
    "hbt_assignment_policy": "place_opt",
    "global_routing_ripup_policy": "routing",
    "global_routing_maze_cost": "routing",
    "global_routing_iteration_controller": "routing",
}

# The original paired-seed matrix contract remains explicit. The two new GNN
# tasks currently bind one fixed seed and do not define matrix execution.
MATRIX_TASK_DIRECTORIES = (
    "tp_gnn_post_placement_timing",
    "placement_initialization",
    "placement_preconditioner",
    "placement_parameters_reference_20260923",
    "hbt_assignment_policy",
    "global_routing_ripup_policy",
    "global_routing_maze_cost",
    "global_routing_iteration_controller",
)

# Names are retained only to explain removal and to guard against accidental
# reintroduction. They are not aliases or executable fallback definitions.
RETIRED_TASK_DIRECTORIES = (
    "netlist_partition", "tp_gnn_learning_objective",
    "tp_gnn_learning_objective_reference_20260923", "tp_gnn_ta3d_learning_objective",
    "tp_gnn_pin_cut_learning_objective", "tp_gnn_bin_timing",
    "placement_init_preconditioner", "placement_init_preconditioner_reference_20260923",
    "placement_parameters", "global_routing_parameters", "global_routing_net_ordering",
)
