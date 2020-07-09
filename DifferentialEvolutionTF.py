import numpy as np
import matplotlib.pyplot as plt


from Solution.DE_TF import DE as DETF
from Solution.DE_TF import Evaluate

from Dataset.Simulation.GaussCurve_TF import FBG_spectra, GaussCurve
from Dataset.loader import DATASET_5fbg_1 as Dataset

import tensorflow as tf

print('loading dataset')
dataset = tf.constant(Dataset(), dtype=tf.dtypes.float32)
print('dataset load done')

I = [5.685, 2.919, 2.25, 0.9342, 0.4047]
W = tf.constant([0.25, 0.25, 0.25, 0.25, 0.25])
ITERATION = 500

detf = DETF()


def init(de):
    de.minx = 1545.0
    de.maxx = 1549.0
    de.I = I
    de.W = W
    de.NP = 50
    de.CR = 0.5
    de.F = 1.1


init(detf)


def plot(data, X):

    simulation = FBG_spectra(data[0], X, tf.repeat([I], X.shape[0], axis=0), W)

    plt.plot(data[0], tf.transpose(simulation), c='gray')
    plt.plot(*data, c='red')

    # A = simulation[0]
    # B = data[1]
    # plt.twinx()
    # plt.plot(data[0], (A-B)**2, c='green')
    # plt.twinx()
    # plt.plot(data[0], ((A-B)**2)/(A+B), c='orange')


def plotPause(info):

    i, data, X, V, dx, dv = info

    if i % 10 == 0:
        plt.clf()
        plot(data, X)
        plt.pause(0.02)


# detf.run(dataset[2], iterations=ITERATION, forEach=plotPause)
# plt.show()


class Plotter():
    def __init__(self):
        self.log = []

    def logger(self, info):
        i, data, X, V, dx, dv = info
        self.log.append(np.min([dx, dv]))


X_log = []
# max_log = []
# min_log = []


def evaluateData(data):
     # data = dataset[10]
    print('run ------------------')
    # print(data)
    X = detf(data, iterations=ITERATION)

    X_log.append(X[tf.argmin(Evaluate(data, X, I, W))])
    plt.clf()

    plt.axhline(1546.923, linestyle=":")
    plt.axhline(1547.29875, linestyle=":")
    plt.axhline(1547.65375, linestyle=":")
    plt.axhline(1548.015, linestyle=":")

    # max_xn, min_xn = computeRange(data, I)

    # max_log.append(max_xn)
    # min_log.append(min_xn)

    plt.plot(X_log)

    plt.pause(0.01)
    return X


Xs = tf.map_fn(evaluateData, dataset)

# for i in range(len(I))[::-1]:
#     plt.fill_between(range(len(X_log)), np.array(
#         min_log)[:, i], np.array(max_log)[:, i], alpha=0.3)

print(Xs)
plt.show()
