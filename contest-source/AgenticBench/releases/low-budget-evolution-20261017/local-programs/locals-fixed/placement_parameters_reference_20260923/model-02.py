# EVOLVE-BLOCK-START
def adjust_learning_rate(native_step_size, step_num, log_hpwl, log_hpwl_prev,
                         overflow_by_die, log_lambda_by_die, step_multiplier_prev,
                         log_gradient_norm_by_die, die_multipliers_prev):
    # Slow the higher-overflow die while retaining bounded shared feedback.
    hpwl_delta = log_hpwl - log_hpwl_prev
    trend = math.tanh(500.0 * hpwl_delta)
    stall = math.exp(-min(20.0, 500.0 * abs(hpwl_delta)))
    overflow_max = max(overflow_by_die)
    overflow_excess = max(0.0, overflow_max - 0.01)

    target_log = (0.08 * stall
                  + 0.04 * math.tanh(20.0 * overflow_excess)
                  - 0.22 * max(0.0, trend))
    target_log = max(-0.22, min(0.12, target_log))
    previous_log = math.log(max(0.0001, min(100.0,
                                            step_multiplier_prev)))
    shared_log = 0.75 * previous_log + 0.25 * target_log
    shared = math.exp(max(-0.25, min(0.14, shared_log)))

    overflow_gap = max(-0.1, min(0.1,
                                  overflow_by_die[0] - overflow_by_die[1]))
    gradient_gap = max(-2.0, min(2.0,
                                  log_gradient_norm_by_die[0]
                                  - log_gradient_norm_by_die[1]))
    die_log = -4.0 * overflow_gap - 0.05 * gradient_gap
    die_log = max(-0.25, min(0.25, die_log))
    return shared, (math.exp(die_log), math.exp(-die_log))
# EVOLVE-BLOCK-END