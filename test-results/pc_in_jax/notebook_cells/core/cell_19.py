def batched_energy_fn(model, activities, input_, output, energy_fn_, record_aux_details=True):
    # vmap a single-example energy_fn over the batch dimension
    return jax.vmap(energy_fn_, in_axes=(None, 0, 0, 0, None))(
        model, activities, input_, output, record_aux_details)


def batched_energy_sum(model, activities, input_, output, energy_fn_, record_aux_details=True):
    # sum-reduce per-example energies (preserves per-example activity grads)
    per_ex, aux = batched_energy_fn(model, activities, input_, output, energy_fn_, record_aux_details)
    return jnp.sum(per_ex), jax.tree_util.tree_map(jnp.sum, aux)


def batched_energy_mean(model, activities, input_, output, energy_fn_, record_aux_details=True):
    # mean-reduce per-example energies (batch-size invariant param grads)
    per_ex, aux = batched_energy_fn(model, activities, input_, output, energy_fn_, record_aux_details)
    return jnp.mean(per_ex), jax.tree_util.tree_map(jnp.mean, aux)