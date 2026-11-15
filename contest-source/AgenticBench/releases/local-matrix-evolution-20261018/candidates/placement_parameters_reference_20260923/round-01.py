# EVOLVE-BLOCK-START
def adjust_learning_rate(native_step_size, step_num, log_hpwl, log_hpwl_prev,
                         overflow_by_die, log_lambda_by_die, step_multiplier_prev,
                         log_gradient_norm_by_die, die_multipliers_prev):
    # Smooth global progress control and gently favor the more congested die.
    hpwl_delta = log_hpwl - log_hpwl_prev
    overflow_max = max(0.0, max(overflow_by_die))
    gradient_skew = math.tanh(
        0.25 * (log_gradient_norm_by_die[0] - log_gradient_norm_by_die[1])
    )

    target_log = (0.06 * math.tanh(10.0 * (overflow_max - 0.02))
                  - 0.08 * math.tanh(250.0 * hpwl_delta)
                  - 0.02 * abs(gradient_skew))
    target_shared = math.exp(target_log)
    shared = 0.85 * step_multiplier_prev + 0.15 * target_shared
    shared = min(1.25, max(0.75, shared))

    overflow_sum = max(0.02, overflow_by_die[0] + overflow_by_die[1])
    overflow_skew = ((overflow_by_die[0] - overflow_by_die[1]) /
                     overflow_sum)
    lambda_skew = math.tanh(
        0.5 * (log_lambda_by_die[0] - log_lambda_by_die[1])
    )
    die_log = (0.12 * overflow_skew
               - 0.03 * gradient_skew
               - 0.02 * lambda_skew)
    die_log = min(0.12, max(-0.12, die_log))

    return shared, (math.exp(die_log), math.exp(-die_log))
# EVOLVE-BLOCK-END