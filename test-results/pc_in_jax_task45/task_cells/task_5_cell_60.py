import numpy as np


def fetch_task4_results(*, entity=None, project=None, group=None):
    api = wandb.Api()
    entity = entity or WANDB_ENTITY or api.default_entity
    if not entity:
        raise ValueError("Set WANDB_ENTITY to the username or team owning the project.")
    runs = api.runs(
        f"{entity}/{project or WANDB_PROJECT}",
        filters={
            "group": group or TASK4_GROUP,
            "state": "finished",
            "config.n_train_iters": 1000,
        },
        order="-created_at",
    )
    newest_results = {}
    for run in runs:
        config = run.config
        parameter = config.get("sweep_parameter")
        accuracy = run.summary.get("test/accuracy_at_1000")
        if parameter not in TASK4_SWEEPS or accuracy is None:
            continue
        if run.summary.get("train/iterations_completed") != 1000:
            continue
        # Keep each comparison at the configured baseline and seed set.
        if (config.get(parameter) not in TASK4_SWEEPS[parameter]
                or config.get("seed") not in TASK4_SEEDS):
            continue
        if any(config.get(name) != value
               for name, value in TASK4_BASE_CONFIG.items()
               if name not in ("seed", parameter)):
            continue
        try:
            accuracy = float(accuracy)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(accuracy) or not 0.0 <= accuracy <= 100.0:
            continue
        key = (parameter, config[parameter], config["seed"])
        newest_results.setdefault(key, {
            "sweep_parameter": parameter,
            "value": config[parameter],
            "seed": config["seed"],
            "test_accuracy": float(accuracy),
            "runtime_seconds": run.summary.get("runtime/seconds"),
            "run_id": run.id,
        })
    if not newest_results:
        raise ValueError(
            "No completed 1,000-iteration results found. Check the entity, "
            "project, and group, and sync any offline runs first."
        )
    expected = {
        (parameter, value, seed)
        for parameter, values in TASK4_SWEEPS.items()
        for value in values
        for seed in TASK4_SEEDS
    }
    missing = expected - newest_results.keys()
    if missing:
        print("Missing or failed (parameter, value, seed):", sorted(missing))
    return [newest_results[key] for key in sorted(newest_results)]


def plot_task4_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, parameter in zip(axes, TASK4_SWEEPS):
        by_value = {}
        for result in results:
            if result["sweep_parameter"] == parameter:
                by_value.setdefault(result["value"], []).append(result["test_accuracy"])
        values = sorted(by_value)
        means = [np.mean(by_value[value]) for value in values]
        stds = [np.std(by_value[value], ddof=1) if len(by_value[value]) > 1 else 0.0
                for value in values]
        if values:
            ax.errorbar(values, means, yerr=stds, marker="o", capsize=4)
        else:
            ax.text(0.5, 0.5, "No completed runs", ha="center", transform=ax.transAxes)
        ax.set(xlabel=parameter, ylabel="Test accuracy at 1,000 iterations (%)")
        ax.set_title(
            "Activity LR fixed at " + str(TASK4_BASE_CONFIG["activity_lr"])
            if parameter == "inference_steps" else
            "Inference steps fixed at " + str(TASK4_BASE_CONFIG["inference_steps"])
        )
        if parameter == "activity_lr":
            ax.set_xscale("log")
        ax.set_xticks(TASK4_SWEEPS[parameter])
        ax.set_xticklabels([str(value) for value in TASK4_SWEEPS[parameter]])
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("task4_wandb_accuracy.png", dpi=180, bbox_inches="tight")
    plt.show()
    return fig


task4_api_results = fetch_task4_results()
for row in task4_api_results:
    print(row)
task4_figure = plot_task4_results(task4_api_results)
