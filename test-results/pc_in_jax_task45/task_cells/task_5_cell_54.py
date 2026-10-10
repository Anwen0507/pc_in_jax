from datetime import datetime, timezone
from uuid import uuid4
import wandb

WANDB_PROJECT = "pc-in-jax-task4"
WANDB_ENTITY = None  # or "your-username-or-team"
WANDB_MODE = "online"  # use "offline" to save locally and sync later
TASK4_GROUP = (
    f"task4-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
)
print("Experiment group:", TASK4_GROUP)


def run_task4_wandb(config, *, mode=None, group=None):
    run_name = (
        f"{config.get('sweep_parameter', 'baseline')}-"
        f"T{config['inference_steps']}-lr{config['activity_lr']:g}-"
        f"{config['activity_optimizer']}-seed{config['seed']}"
    )
    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=group or TASK4_GROUP,
        name=run_name,
        job_type="task4",
        tags=["task4", config.get("sweep_parameter", "baseline")],
        config=dict(config),
        mode=mode or WANDB_MODE,
        dir=".",
    ) as run:
        run.define_metric("train/iteration")
        run.define_metric("train/energy", step_metric="train/iteration")
        run.define_metric("test/accuracy", step_metric="train/iteration")
        result = run_task4_experiment(config, metrics_callback=run.log)
        run.summary.update({
            "test/final_accuracy": result["final_test_accuracy"],
            "train/iterations_completed": result["history"][-1]["train/iteration"],
            "runtime/seconds": result["runtime_seconds"],
        })
        if (config["n_train_iters"] == 1000
                and result["history"][-1]["train/iteration"] == 1000):
            run.summary["test/accuracy_at_1000"] = result["final_test_accuracy"]
        result["run_id"] = run.id
        result["entity"] = run.entity
        result["project"] = run.project
        # The context manager finishes and flushes this run before the next.
    return result
