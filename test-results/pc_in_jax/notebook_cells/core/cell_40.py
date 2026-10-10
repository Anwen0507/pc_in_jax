@eqx.filter_jit
def train_step(model, input_, output, p_optim, p_opt_state, x_optim, T, record_aux_details=True):
    # one batch: feedforward-initialize, settle for T steps, update the parameters once

    # inference: initialize the activities per example (vmap over the batch)...
    activities = jax.vmap(init_activities_with_ffwd, in_axes=(None, 0))(model, input_)
    x_opt_state = x_optim.init(activities)

    # ...then relax them for T steps
    def settle_body(carry, _):
        activities, x_opt_state = carry
        activities, x_opt_state, aux = update_activities(
            model, activities, x_optim, x_opt_state, input_, output, record_aux_details)
        return (activities, x_opt_state), aux

    (activities, _), aux_scan = jax.lax.scan(
        settle_body, (activities, x_opt_state), None, length=T)

    # energy of the settled state, i.e. what the parameter update sees
    energy, _ = batched_energy_mean(model, activities, input_, output, energy_fn, False)

    # learning: one parameter update at the settled activities
    model, p_opt_state, _ = update_params(
        model, activities, p_optim, p_opt_state, input_, output, record_aux_details=False)

    return {
        "model": model,
        "p_opt_state": p_opt_state,
        "activities": activities,
        "energy": energy,
        "aux": aux_scan,  # per-layer energies along the settling trajectory, each of shape (T,)
    }