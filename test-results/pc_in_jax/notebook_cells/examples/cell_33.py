param_grads, aux = compute_param_grad(model, activities, img_batch, label_batch)
for l, layer_grad in enumerate(param_grads.layers):
    linear_grad = layer_grad.layers[1]  # Sequential([activation, Linear]) -> the Linear
    print(f"layer {l}: dE/dW {linear_grad.weight.shape}, dE/db {linear_grad.bias.shape}, "
          f"mean |dE/dW| = {jnp.abs(linear_grad.weight).mean():.2e}")