# for activity_optimizer in ("sgd", "adam"):
#     run_task4_wandb(
#         {**TASK4_BASE_CONFIG, "activity_optimizer": activity_optimizer,
#          "sweep_parameter": "activity_optimizer"},
#         group=f"{TASK4_GROUP}-optimizer-comparison",
#     )
