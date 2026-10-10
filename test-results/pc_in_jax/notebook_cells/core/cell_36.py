@eqx.filter_jit
def update_activities(model, activities, x_optim, x_opt_state, input_, output, record_aux_details=True):
    activity_grads, aux = compute_activity_grad(model, activities, input_, output, record_aux_details)
    updates, x_opt_state = x_optim.update(activity_grads, x_opt_state, params=activities)
    activities = eqx.apply_updates(activities, updates)
    return activities, x_opt_state, aux


# note that the only difference with the above function is...
# ...the variable we are updating (parameters vs. activities)
@eqx.filter_jit
def update_params(model, activities, p_optim, p_opt_state, input_, output, record_aux_details=True):
    param_grads, aux = compute_param_grad(model, activities, input_, output, record_aux_details)
    updates, p_opt_state = p_optim.update(param_grads, p_opt_state, params=model)
    model = eqx.apply_updates(model, updates)
    return model, p_opt_state, aux