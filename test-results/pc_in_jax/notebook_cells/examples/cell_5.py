SEED = 42

INPUT_DIM, OUTPUT_DIM = 28 * 28, 10
HIDDEN_SIZE = 300
N_LAYERS = 3  # number of weight layers

ACTIVITY_LR = 0.1 # settling / inference learning rate
INFERENCE_STEPS = 20 # number of inference steps per batch

PARAM_LR = 1e-3 # parameter learning rate
BATCH_SIZE = 64

TEST_EVERY = 50 # evaluate every 50 batches
N_TRAIN_ITERS = 500