from time import perf_counter

TASK4_BASE_CONFIG = {
    "seed": SEED,
    "input_dim": INPUT_DIM,
    "output_dim": OUTPUT_DIM,
    "hidden_size": HIDDEN_SIZE,
    "n_layers": N_LAYERS,
    "activation": "tanh",
    "activity_lr": ACTIVITY_LR,
    "inference_steps": INFERENCE_STEPS,
    "param_lr": PARAM_LR,
    "batch_size": BATCH_SIZE,
    "test_every": TEST_EVERY,
    "n_train_iters": 1000,
    "activity_optimizer": "sgd",
}
TASK4_SWEEPS = {
    "inference_steps": [1, 5, 20, 50, 100],
    "activity_lr": [0.001, 0.01, 0.1, 0.5, 1.0],
}
TASK4_SEEDS = [SEED]


def task4_configs():
    for parameter, values in TASK4_SWEEPS.items():
        for value in values:
            for seed in TASK4_SEEDS:
                yield {
                    **TASK4_BASE_CONFIG,
                    parameter: value,
                    "seed": seed,
                    "sweep_parameter": parameter,
                }


def run_task4_experiment(config, metrics_callback=None):
    # Never reuse trained_model from the earlier demonstration.
    torch.manual_seed(config["seed"])
    experiment_model = FeedforwardPCModel(
        jr.PRNGKey(config["seed"]),
        input_size=config["input_dim"],
        hidden_size=config["hidden_size"],
        output_size=config["output_dim"],
        activation_fn=jax.nn.tanh,
        n_layers=config["n_layers"],
    )
    started = perf_counter()
    _, history = train(
        model=experiment_model,
        activity_lr=config["activity_lr"],
        inference_steps=config["inference_steps"],
        param_lr=config["param_lr"],
        batch_size=config["batch_size"],
        test_every=config["test_every"],
        n_train_iters=config["n_train_iters"],
        seed=config["seed"],
        activity_optimizer=config["activity_optimizer"],
        metrics_callback=metrics_callback,
        return_history=True,
    )
    if not history or history[-1]["train/iteration"] != config["n_train_iters"]:
        raise ValueError("Training did not complete the requested number of iterations.")
    return {
        "config": dict(config),
        "history": history,
        "final_test_accuracy": history[-1]["test/accuracy"],
        "runtime_seconds": perf_counter() - started,
    }
