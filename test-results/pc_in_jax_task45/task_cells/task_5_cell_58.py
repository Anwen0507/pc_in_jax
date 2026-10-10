if WANDB_MODE == "online":
    wandb.login()

wandb_results = []
failed_experiments = []
for config in task4_configs():
    parameter = config["sweep_parameter"]
    print(f"\nRunning {parameter}={config[parameter]}, seed={config['seed']}")
    try:
        result = run_task4_wandb(config)
    except FloatingPointError as error:
        failed_experiments.append({"config": config, "error": str(error)})
        print(f"Failed: {error}")
        continue
    wandb_results.append(result)
    print(f"Run {result['run_id']}: {result['final_test_accuracy']:.2f}% "
          f"in {result['runtime_seconds']:.1f} seconds")
print(f"\nCompleted {len(wandb_results)} runs; {len(failed_experiments)} failed.")
