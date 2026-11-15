from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from local_bench.program_safety import read_repo_file, single_definition, unified_diff, validate_no_external_effects
from local_tasks.placement_control_language import _is_bounded_iteration, SAFE_DIRECT_CALLS
from local_bench.numerical_programs import validate_helpers

CONCURRENT = "Place-3D/src/placement/concurrent_dreamplace.py"
NONLINEAR = "Place-3D/thirdparty/DREAMPlace/dreamplace/NonLinearPlace.py"
NESTEROV = "Place-3D/thirdparty/DREAMPlace/dreamplace/NesterovAcceleratedGradientOptimizer.py"
CONTROLLER_ARGS = [
    "native_step_size", "step_num", "log_hpwl", "log_hpwl_prev",
    "overflow_by_die", "log_lambda_by_die", "step_multiplier_prev",
    "log_gradient_norm_by_die", "die_multipliers_prev",
]


def _replace_once(source, needle, replacement, label):
    if source.count(needle) != 1:
        raise ValueError("Nesterov reference-20260923 anchor {} must occur once".format(label))
    return source.replace(needle, replacement, 1)


def _validate_controller(node):
    helper_names = validate_helpers(node)
    args = node.args
    if ([item.arg for item in args.args] != CONTROLLER_ARGS or args.posonlyargs
            or args.kwonlyargs or args.vararg or args.kwarg or args.defaults
            or args.kw_defaults or node.decorator_list):
        raise ValueError("adjust_learning_rate must keep the Nesterov nine-argument signature")
    for child in ast.walk(node):
        if child is not node and isinstance(child, (ast.AsyncFor, ast.While, ast.Lambda, ast.ClassDef)):
            raise ValueError("Nesterov controller contains forbidden syntax")
        if isinstance(child, ast.For) and not _is_bounded_iteration(child.iter):
            raise ValueError("Nesterov controller loops must have a bounded two-die iterator")
        if isinstance(child, ast.comprehension) and (child.is_async or not _is_bounded_iteration(child.iter)):
            raise ValueError("Nesterov controller comprehensions must have a bounded two-die iterator")
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id not in SAFE_DIRECT_CALLS | helper_names:
            raise ValueError("Nesterov controller calls unsupported function {}".format(child.func.id))
        if isinstance(child, ast.Attribute):
            root = child.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id != "math":
                raise ValueError("Nesterov controller may access attributes only through math")
    validate_no_external_effects(node, allowed_attribute_roots={"math"})


def _patch_optimizer(before):
    """Control native no-BB backtracking without changing its state updates.

    Call once before the backtracking loop and hold the scale across trials.
    The callback is not a post-step coordinate adjustment. Native alpha updates,
    max_backtrack_cnt=10, acceptance and acceleration are untouched. Identity
    takes the original arithmetic branch verbatim.
    """
    after = _replace_once(before, "        self.use_bb = use_bb\n",
                          "        self.use_bb = use_bb\n        self.agentic3d_step_scale = None\n", "hook initialization")
    after = _replace_once(after, "    def step(self, closure=None):\n        if self.use_bb:\n",
                          "    def step(self, closure=None):\n"
                          "        if self.agentic3d_step_scale is not None and self.use_bb:\n"
                          "            raise ValueError(\"Reference Nesterov step controller requires native no-BB backtracking\")\n"
                          "        if self.use_bb:\n", "native no-BB dispatch")
    anchor = "                #ttt = time.time()\n                while True:\n"
    controlled = """                # Decide once; retain the same factors through native backtracking.
                agentic3d_scale = None
                if self.agentic3d_step_scale is not None:
                    with torch.no_grad():
                        agentic3d_scale = self.agentic3d_step_scale(alpha_k, g_k.detach())
                    if agentic3d_scale is not None:
                        if (agentic3d_scale.shape != g_k.shape
                                or agentic3d_scale.device != g_k.device
                                or agentic3d_scale.dtype != g_k.dtype
                                or not bool(torch.isfinite(agentic3d_scale).all().item())
                                or not bool((agentic3d_scale > 0).all().item())):
                            raise ValueError("Invalid Nesterov coordinate step scale")
"""
    after = _replace_once(after, anchor, controlled + anchor, "pre-backtracking control")
    return _replace_once(after, "                    u_kp1 = v_k - alpha_k*g_k\n",
                          "                    if agentic3d_scale is None:\n"
                          "                        u_kp1 = v_k - alpha_k*g_k\n"
                          "                    else:\n"
                          "                        u_kp1 = v_k - alpha_k*(g_k*agentic3d_scale)\n",
                          "native no-BB gradient step")


def _patch_nonlinear(before, controller):
    runtime = Path(__file__).with_name("nesterov_runtime.py").read_text(encoding="utf-8")
    # Keep the runtime's imports/module docstring out of the injected body.
    runtime_tree = ast.parse(runtime)
    definitions = "\n\n".join(ast.get_source_segment(runtime, node)
                               for node in runtime_tree.body if isinstance(node, ast.FunctionDef))
    after = _replace_once(before, "import inspect\n",
                          "import inspect\nimport math\n\n" + definitions + "\n\n" + controller + "\n",
                          "controller definitions")
    after = _replace_once(after, "                initial_state = copy.deepcopy(optimizer.state_dict())\n",
                          "                initial_state = copy.deepcopy(optimizer.state_dict())\n"
                          "                if optimizer_name.lower() != \"nesterov\" or optimizer.use_bb:\n"
                          "                    raise ValueError(\"Reference Nesterov controller requires native no-BB Nesterov\")\n"
                          "                agentic3d_nesterov_state = {}\n", "controller state")
    anchor = "                    # plot placement\n                    if params.plot_flag and (iteration % 100 == 0 or iteration == 999):\n"
    hook = textwrap.indent(textwrap.dedent('''\
        def agentic3d_step_scale(native_step_size, gradient):
            return _agentic3d_nesterov_step_control(
                adjust_learning_rate, placedb, model, pos, cur_metric, iteration,
                agentic3d_nesterov_state, native_step_size, gradient,
            )
        optimizer.agentic3d_step_scale = agentic3d_step_scale

        '''), "                    ")
    return _replace_once(after, anchor, hook + anchor, "per-step Nesterov callback")


def materialize(program_text: str, open3dbench_root: Path) -> dict:
    controller, node = single_definition(program_text, ast.FunctionDef, "adjust_learning_rate")
    _validate_controller(node)
    controller = "\n".join(line.rstrip() for line in controller.splitlines()).rstrip() + "\n"
    concurrent = read_repo_file(open3dbench_root, CONCURRENT)
    if concurrent.count('        "optimizer": "nesterov",\n') != 1:
        raise ValueError("reference-20260923 requires the unchanged native Nesterov selector")
    nonlinear = read_repo_file(open3dbench_root, NONLINEAR)
    nesterov = read_repo_file(open3dbench_root, NESTEROV)
    nonlinear_after = _patch_nonlinear(nonlinear, controller)
    nesterov_after = _patch_optimizer(nesterov)
    compile(nonlinear_after, NONLINEAR, "exec")
    compile(nesterov_after, NESTEROV, "exec")
    return {
        "schema_version": 1,
        "config_overrides": {"placement": {"target_density": 0.6}},
        "candidate_files": {"placement.json": {"concurrent_dreamplace": {
            "cross_die_net_weight": 1.0, "iterations": 4000, "num_bins": 512,
            "target_density": 0.6, "stop_overflow": 0.01, "density_weight": 0.00003,
            "macro_place_flag": True, "macro_halo_x": 10.0, "macro_halo_y": 10.0,
            "learning_rate": 0.01, "die_params": {}, "hide_fillers_in_plot": False,
        }}},
        "source_patch": unified_diff(NONLINEAR, nonlinear, nonlinear_after)
                        + unified_diff(NESTEROV, nesterov, nesterov_after),
        "notes": "Reference-20260923 native no-BB Nesterov gradient-step control; identity is 1 x (1,1). New native qualification required.",
    }
