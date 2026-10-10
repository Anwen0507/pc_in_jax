trained_model = train(
    model=model,
    activity_lr=ACTIVITY_LR,
    inference_steps=INFERENCE_STEPS,
    param_lr=PARAM_LR,
    batch_size=BATCH_SIZE,
    test_every=TEST_EVERY,
    n_train_iters=N_TRAIN_ITERS
)