from src.cnn_builder import Layer

DROPOUT_PROB = 1
NETWORK_STRUCTURE = [Layer("dense", [28*28*1, 256]),
                     Layer("dropout", [DROPOUT_PROB]),
                     Layer("dense", [256, 10])]
NETWORK_ANCHOR = -2
NETWORK_PATH = "model/mnist/model_3.ckpt"
INIT = 1e-1
LEARNING_RATE = 1e-2
DISTRIBUTION_PATH = "profile/mnist/model3_mdist_distribution.pkl"