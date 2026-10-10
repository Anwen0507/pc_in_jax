@eqx.filter_jit
def predict(model, input_):
    return jax.vmap(model)(input_)


def evaluate(model, test_loader):
    # Count examples rather than averaging batch accuracies: the last batch
    # may be smaller, and every test image should have the same weight.
    n_correct, n_examples = 0, 0
    for img_batch, label_batch in test_loader:
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
        preds = predict(model, img_batch)
        if not bool(jnp.all(jnp.isfinite(preds))):
            raise FloatingPointError("Non-finite predictions; this configuration diverged.")
        correct = jnp.argmax(label_batch, axis=1) == jnp.argmax(preds, axis=1)
        n_correct += int(jax.device_get(jnp.sum(correct)))
        n_examples += len(img_batch)
    if n_examples == 0:
        raise ValueError("The test loader is empty.")
    return 100.0 * n_correct / n_examples


def train(model, activity_lr, inference_steps, param_lr, batch_size,
          test_every, n_train_iters, *, seed=SEED, activity_optimizer="sgd",
          metrics_callback=None, return_history=False):
    if n_train_iters < 1 or test_every < 1 or inference_steps < 1:
        raise ValueError("n_train_iters, test_every, and inference_steps must be positive.")
    if activity_optimizer == "sgd":
        x_optim = optax.sgd(activity_lr)
    elif activity_optimizer == "adam":
        x_optim = optax.adam(activity_lr)
    else:
        raise ValueError("activity_optimizer must be 'sgd' or 'adam'.")
    p_optim = optax.adam(param_lr)
    p_opt_state = p_optim.init(eqx.filter(model, eqx.is_array))

    train_loader, test_loader = get_mnist_loaders(batch_size, seed=seed)
    train_loader = infinite_loader(train_loader)
    history = []
    for train_iter, (img_batch, label_batch) in enumerate(train_loader, start=1):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
        out = train_step(model,
                         img_batch,
                         label_batch,
                         p_optim,
                         p_opt_state,
                         x_optim,
                         T=inference_steps,
                         record_aux_details=False)
        model, p_opt_state = out["model"], out["p_opt_state"]

        # Always evaluate after the final parameter update, even when the
        # requested iteration count is not a multiple of test_every.
        if train_iter % test_every == 0 or train_iter == n_train_iters:
            train_energy = float(jax.device_get(out["energy"]))
            if not bool(jnp.isfinite(train_energy)):
                raise FloatingPointError("Non-finite training energy; this configuration diverged.")
            avg_test_acc = evaluate(model, test_loader)
            metrics = {
                "train/iteration": train_iter,
                "train/energy": train_energy,
                "test/accuracy": avg_test_acc,  # percentage, from 0 to 100
            }
            history.append(metrics)
            if metrics_callback is not None:
                metrics_callback(metrics)
            print(
                f"Train iter {train_iter}, train energy={train_energy:.4f}, "
                f"avg test accuracy={avg_test_acc:.2f}%")
        if train_iter >= n_train_iters:
            break

    return (model, history) if return_history else model


def infinite_loader(dataloader):
    while True:
        yielded = False
        for batch in dataloader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("The training loader is empty.")
