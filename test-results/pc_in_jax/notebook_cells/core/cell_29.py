# "the gradient of the (summed) energy with respect to the 2nd argument (the activities)"
def compute_activity_grad(model, activities, input_, output, record_aux_details=True):
    # per-example activity gradient of the batch-SUMMED energy
    return jax.grad(batched_energy_sum, argnums=1, has_aux=True)(
        model, activities, input_, output, energy_fn, record_aux_details)


# for parameters, the only differences are: we differentiate the batch-MEAN energy...
# ...with respect to the 1st argument (the model), skipping its non-array leaves

def compute_param_grad(model, activities, input_, output, record_aux_details=True):
    # batch-averaged parameter gradient
    return eqx.filter_grad(batched_energy_mean, has_aux=True)(
        model, activities, input_, output, energy_fn, record_aux_details)