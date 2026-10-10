dEdzs, aux = compute_activity_grad(model, activities, img_batch, label_batch)
for l, dEdz in enumerate(dEdzs):
    print(f"activity gradient dE/dz_{l}: shape {dEdz.shape}, mean |dE/dz| = {jnp.abs(dEdz).mean():.4f}")