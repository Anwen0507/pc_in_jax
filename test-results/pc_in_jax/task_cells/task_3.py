settling_steps = 50
x_optim = optax.sgd(ACTIVITY_LR)
p_optim = optax.adam(PARAM_LR)
p_opt_state = p_optim.init(eqx.filter(model, eqx.is_array))

settling_out = train_step(
    model, img_batch, label_batch, p_optim, p_opt_state, x_optim,
    T=settling_steps, record_aux_details=True,
)

fig, ax = plt.subplots(figsize=(8, 5))
for name, energies in settling_out["aux"].items():
    ax.plot(range(settling_steps), energies, label=name.removeprefix("energy/"))
ax.set(
    xlabel="Settling step (before update)",
    ylabel="Energy (sum over batch)",
    title="Per-layer energies over 50 settling steps",
)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()