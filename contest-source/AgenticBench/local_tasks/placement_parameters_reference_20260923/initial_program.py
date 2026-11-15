# EVOLVE-BLOCK-START
def adjust_learning_rate(native_step_size, step_num, log_hpwl, log_hpwl_prev,
                         overflow_by_die, log_lambda_by_die, step_multiplier_prev,
                         log_gradient_norm_by_die, die_multipliers_prev):
    """Identity control: retain native Nesterov backtracking and acceleration."""
    return 1.0, (1.0, 1.0)
# EVOLVE-BLOCK-END
