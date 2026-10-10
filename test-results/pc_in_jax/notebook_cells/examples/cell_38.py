x_optim = optax.sgd(ACTIVITY_LR)
x_opt_state = x_optim.init(activities)

settled = activities
for step in range(5):
    settled, x_opt_state, _ = update_activities(
        model, settled, x_optim, x_opt_state, img_batch, label_batch)
    total, per_layer = batched_energy_mean(model, settled, img_batch, label_batch, energy_fn)
    print(f"step {step + 1}: total energy {total:.4f}  |  "
          + "  ".join(f"{k.split('/')[1]}: {v:.2e}" for k, v in per_layer.items()))