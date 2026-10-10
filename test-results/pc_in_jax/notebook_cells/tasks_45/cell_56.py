smoke_config = {
    **TASK4_BASE_CONFIG,
    "sweep_parameter": "smoke",
    "inference_steps": 1,
    "n_train_iters": 2,
    "test_every": 2,
}
smoke_result = run_task4_wandb(
    smoke_config, mode="offline", group=f"{TASK4_GROUP}-smoke",
)
