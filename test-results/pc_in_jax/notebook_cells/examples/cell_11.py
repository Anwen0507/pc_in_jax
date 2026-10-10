key = jr.PRNGKey(SEED)

model = FeedforwardPCModel(
    key,
    input_size=INPUT_DIM,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_DIM,
    activation_fn=jax.nn.tanh,
    n_layers=N_LAYERS,
)
model