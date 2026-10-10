# a single example
activities_one = init_activities_with_ffwd(model, img_batch[0])
for l, z in enumerate(activities_one):
    print(f"single example, activity z_{l}: {z.shape}")

# the whole batch
activities = jax.vmap(init_activities_with_ffwd, in_axes=(None, 0))(model, img_batch)
for l, z in enumerate(activities):
    assert z.shape == (img_batch.shape[0],) + activities_one[l].shape
    print(f"batch, activity z_{l}: {z.shape}")