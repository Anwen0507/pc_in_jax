class FeedforwardPCModel(eqx.Module):
    # Each layer is Sequential([activation, Linear]): a PRE-activation layer.
    layers: list[eqx.Module]

    def __init__(
            self,
            key,
            *,
            input_size,
            hidden_size,
            output_size,
            activation_fn,
            input_activation_fn=eqx.nn.Identity(),
            n_layers=2,
            use_bias=True,
    ):
        keys = jr.split(key, n_layers)
        widths = [input_size] + [hidden_size] * (n_layers - 1) + [output_size]

        layers = []
        for i in range(n_layers):
            act = input_activation_fn if i == 0 else activation_fn
            act_layer = act if isinstance(
                act, eqx.Module) else _StaticActivation(act)
            linear = eqx.nn.Linear(widths[i],
                                   widths[i + 1],
                                   key=keys[i],
                                   use_bias=use_bias)
            layers.append(eqx.nn.Sequential([act_layer, linear]))

        self.layers = layers

    def __call__(self, x):
        # feedforward pass for a SINGLE example
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i):
        # Implements indexing so PC loops can write
        # err = activities[l] - model[l](activities[l - 1]).
        return self.layers[i]

    def __len__(self):
        return len(self.layers)


class _StaticActivation(eqx.Module):
    # Not important to understand this class for this tutorial.
    # Wraps a bare callable (e.g. jax.nn.tanh) as an eqx.Module whose function
    # lives in static metadata, not as a pytree leaf.
    fn: Callable = eqx.field(static=True)

    def __call__(self, x, *, key=None):
        return self.fn(x)