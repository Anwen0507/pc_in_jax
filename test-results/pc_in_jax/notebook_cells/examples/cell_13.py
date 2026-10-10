train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)
img_batch, label_batch = next(iter(train_loader))

# we need to turn the torch.Tensor data into numpy arrays for jax
img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
print(f"image batch: {img_batch.shape}, label batch: {label_batch.shape}")

# a single example
print(f"model(x) for one example: {model(img_batch[0]).shape}")

# the whole batch: vmap adds the batch axis for us
print(f"vmap(model)(X) for the batch: {jax.vmap(model)(img_batch).shape}")