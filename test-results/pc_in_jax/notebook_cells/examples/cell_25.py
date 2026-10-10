per_example_energy, _ = batched_energy_fn(model, activities, img_batch, label_batch, energy_fn)
print(f"per-example energies: shape {per_example_energy.shape}, first five: {per_example_energy[:5]}")

energy, aux = batched_energy_mean(model, activities, img_batch, label_batch, energy_fn)
print(f"\nbatch-mean energy: {energy:.4f}")
for name, value in aux.items():
    print(f"  {name}: {value:.4f}")