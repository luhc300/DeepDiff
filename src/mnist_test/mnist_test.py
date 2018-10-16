from keras.datasets import mnist
# from keras.datasets import fashion_mnist
from src.cnn_profiler import CNNProfiler
from src.configs.network_configs.mnist.network_config_3 import NETWORK_STRUCTURE, NETWORK_ANCHOR, NETWORK_PATH, INIT, LEARNING_RATE
from src.distribution import Distribution
import matplotlib.pyplot as plt
import src.augmentation as aug
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.cross_validation import train_test_split, cross_val_score
import numpy as np
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# test_x, test_y= aug.random_pertubate(X_train[:2], y_train[:2], 1)
# plt.subplot(211)
# plt.imshow(X_train[0].reshape(28,28))
# plt.subplot(212)
# plt.imshow(test_x[0].reshape(28,28))
# plt.show()
# X_ptb, y_ptb = aug.random_pertubate(X_train, y_train, 1)
# X_train = np.concatenate((X_train[:1000], X_ptb), axis=0)
# y_train = np.concatenate((y_train[:1000], y_ptb), axis=0)
X_train = X_train.astype("float32")
X_train /= 255
X_test = X_test.astype("float32")
X_test /= 255
cnn_profiler = CNNProfiler(NETWORK_STRUCTURE, network_anchor=NETWORK_ANCHOR, network_path=NETWORK_PATH, init=INIT, lr=LEARNING_RATE)
input_shape = [None, 28, 28, 1]
output_shape = [None, 10]


def train():
    x = X_train.astype("float32")[:400]
    y = y_train.reshape(-1)[:400]
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    cnn_profiler.train(input_shape, output_shape, x, y, iter=2000)


def test():
    x = X_test.astype("float32")
    y = y_test.reshape(-1)
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    acc = 0
    for i in range(0, 8001, 2000):
        acc += cnn_profiler.test([None, 28, 28, 1], [None, 10], x[i:i+2000], y[i:i+2000])
    acc /= 5
    print(acc)


def correct_wrong_mine():
    x = X_test.astype("float32")[:10000]
    y = y_test.reshape(-1)[:10000]
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    correct_mid, wrong_mid = cnn_profiler.get_correct_mid(input_shape, output_shape, x, y, anchor=-2)
    print(len(correct_mid), len(wrong_mid))
    correct_mid_selected = correct_mid[:len(wrong_mid)]
    label_0 = np.zeros(len(wrong_mid))
    label_1 = np.ones(len(correct_mid_selected))
    all_vec = np.concatenate((correct_mid_selected, wrong_mid), axis=0)
    all_label = np.concatenate((label_1, label_0), axis=0)
    shuffle = np.arange(len(all_vec))
    np.random.shuffle(shuffle)
    all_vec = all_vec[shuffle]
    all_label = all_label[shuffle]
    # x_tr, y_tr, x_t, y_t = train_test_split(all_vec, all_label)
    # clf = RandomForestClassifier()
    clf= SVC(C=10)
    scores = cross_val_score(clf, all_vec, all_label, cv=5)
    print(scores)

def correct_mine():
    x = X_test.astype("float32")[:10000]
    y = y_test.reshape(-1)[:10000]
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    correct_mid, wrong_mid = cnn_profiler.get_correct_mid(input_shape, output_shape, x, y, anchor=-2)
    print(len(correct_mid), len(wrong_mid))
    clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)
    clf.fit(correct_mid[:6000])
    result = clf.score_samples(correct_mid[6000:])
    # result = clf.predict(wrong_mid)
    print(len(result), len(result[result==-1]))
# correct_wrong_mine()
correct_mine()
# mnist = read_data_sets("data/MNIST_data/", one_hot=True)
# x = mnist.train.images
# print(x[x>0])
# y = mnist.train.labels
# cnn_profiler.train([None, 28, 28, 1], [None, 10], x, y)
# x = mnist.test.images[:500]
# y = mnist.test.labels[:500]
# cnn_profiler.test([None, 28, 28, 1], [None, 10], x, y)
