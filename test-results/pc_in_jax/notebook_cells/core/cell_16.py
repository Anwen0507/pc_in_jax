def energy_fn(model, activities, input_, output, record_aux_details=True):
    # Layer-wise PC energy for a SINGLE example.
    # Layer 0 predicts from input_; interior layers predict from the previous
    # activity; the final layer's prediction is compared against output.
    last_idx = len(model) - 1
    energies = []
    aux = {}
    for layer_idx, layer in enumerate(model.layers):
        if layer_idx == 0 and layer_idx == last_idx:
            # A one-layer model predicts the clamped target directly.
            pred = layer(input_)
            err = output - pred
            key = 'energy/output'
        elif layer_idx == 0:
            pred = layer(input_)
            err = activities[layer_idx] - pred
            key = f'energy/layer_{layer_idx}'
        elif layer_idx == last_idx:
            pred = layer(activities[layer_idx - 1])
            err = output - pred
            key = 'energy/output'
        else:
            pred = layer(activities[layer_idx - 1])
            err = activities[layer_idx] - pred
            key = f'energy/layer_{layer_idx}'
        energies.append(0.5 * jnp.sum(err ** 2))
        if record_aux_details:
            aux[key] = energies[-1]
    return sum(energies), aux