"""Trusted injected runtime; this file is not part of the evolved program.

The host adapter injects these definitions into NonLinearPlace.py. Torch and
math are provided by that module. No EDA is run by importing this file.
"""
import math


def _agentic3d_project_step_control(decision, previous):
    if not isinstance(decision, (tuple, list)) or len(decision) != 2:
        raise ValueError("Nesterov controller must return (step_multiplier, (bottom, top))")
    shared = float(decision[0])
    if not math.isfinite(shared) or not 0.0001 <= shared <= 100.0:
        raise ValueError("Nesterov step multiplier must be finite and in [0.0001, 100]")
    proposed = decision[1]
    if not isinstance(proposed, (tuple, list)) or len(proposed) != 2:
        raise ValueError("Nesterov die multipliers must contain bottom and top")
    if not isinstance(previous, (tuple, list)) or len(previous) != 2:
        raise ValueError("Nesterov previous die multipliers must contain bottom and top")
    proposed = tuple(float(value) for value in proposed)
    previous = tuple(float(value) for value in previous)
    if any(not math.isfinite(value) or value <= 0 for value in proposed + previous):
        raise ValueError("Nesterov die multipliers must be finite and positive")
    # Die controls specify relative gradient-step scale, not absolute LR.
    # Bound the ratio, remove its common magnitude, then smooth in log space.
    ratio = max(0.25, min(4.0, proposed[0] / proposed[1]))
    root = math.sqrt(ratio)
    target = (root, 1.0 / root)
    mixed = tuple(math.exp(0.8 * math.log(old) + 0.2 * math.log(new))
                  for old, new in zip(previous, target))
    geometric_mean = math.sqrt(mixed[0] * mixed[1])
    return shared, (mixed[0] / geometric_mean, mixed[1] / geometric_mean)


def _agentic3d_nesterov_region_masks(placedb, data, pos):
    if (len(placedb.regions) != 2
            or getattr(placedb, "concurrent_lol_region_names", None) != ["bottom", "top"]):
        raise ValueError("Nesterov controller requires named bottom/top Concurrent LoL regions")
    masks = []
    union = torch.zeros_like(pos, dtype=torch.bool)
    for region in range(2):
        nodes = torch.zeros(placedb.num_nodes, dtype=torch.bool, device=pos.device)
        nodes[:placedb.num_movable_nodes] = (
            data.node2fence_region_map[:placedb.num_movable_nodes] == region
        )
        if placedb.num_filler_nodes:
            starts = placedb.filler_start_map
            if starts is None or len(starts) != 3:
                raise ValueError("Nesterov controller requires exactly two filler ranges")
            if (int(starts[0]) != 0 or int(starts[-1]) != placedb.num_filler_nodes
                    or not 0 <= int(starts[1]) <= placedb.num_filler_nodes):
                raise ValueError("Nesterov filler ranges must cover every filler exactly once")
            base = placedb.num_nodes - placedb.num_filler_nodes
            nodes[base + int(starts[region]):base + int(starts[region + 1])] = True
        mask = torch.cat((nodes, nodes))
        if bool((union & mask).any().item()):
            raise ValueError("Nesterov die masks overlap")
        union |= mask
        masks.append(mask)
    expected_nodes = torch.zeros(placedb.num_nodes, dtype=torch.bool, device=pos.device)
    expected_nodes[:placedb.num_movable_nodes] = True
    if placedb.num_filler_nodes:
        expected_nodes[-placedb.num_filler_nodes:] = True
    if not torch.equal(union, torch.cat((expected_nodes, expected_nodes))):
        raise ValueError("Nesterov die masks must cover all movable and filler coordinates")
    return masks


def _agentic3d_nesterov_step_control(controller, placedb, model, pos, metric,
                                    iteration, state, native_step_size, gradient):
    """Evaluate controls once at v_k before native no-BB backtracking.

    Return None for the exact identity so the original arithmetic expression is
    retained. Otherwise return a coordinate-wise multiplier for the gradient
    step only; no position, gradient, objective, or optimizer state is mutated.
    """
    native_step = float(native_step_size.detach().item())
    if not math.isfinite(native_step) or native_step <= 0:
        raise ValueError("Native Nesterov step hint is non-finite or non-positive")
    if metric.overflow.numel() != 2 or model.density_weight.numel() != 2:
        raise ValueError("Nesterov controller requires two overflow/density values")
    masks = _agentic3d_nesterov_region_masks(placedb, model.data_collections, pos)
    hpwl = max(float(metric.hpwl.detach().sum().item()), 1.0e-30)
    log_hpwl = math.log(hpwl)
    overflow = tuple(float(metric.overflow.detach()[i].item()) for i in range(2))
    log_lambda = tuple(math.log(max(float(model.density_weight.detach()[i].item()), 1.0e-30))
                       for i in range(2))
    log_gradient = tuple(math.log(max(float(gradient.detach()[mask].norm().item()), 1.0e-30))
                         for mask in masks)
    if not all(math.isfinite(value) for value in (log_hpwl,) + overflow + log_lambda + log_gradient):
        raise ValueError("Nesterov controller observations must be finite")
    previous_die = state.get("die", (1.0, 1.0))
    decision = controller(native_step, iteration, log_hpwl, state.get("hpwl", log_hpwl),
                          overflow, log_lambda, state.get("shared", 1.0),
                          log_gradient, previous_die)
    shared, die = _agentic3d_project_step_control(decision, previous_die)
    state.update(hpwl=log_hpwl, shared=shared, die=die, calls=state.get("calls", 0) + 1)
    logging.info("NesterovStepControl step=%d call=%d native_step=%.12g shared=%.12g bottom=%.12g top=%.12g",
                 iteration, state["calls"], native_step, shared, die[0], die[1])
    if shared == 1.0 and die == (1.0, 1.0):
        return None
    scale = torch.ones_like(gradient)
    for region in range(2):
        scale.masked_fill_(masks[region], shared * die[region])
    return scale
