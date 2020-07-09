import numpy as np
import matplotlib.pyplot as plt


from Solution.DESPCM.DESPCM import DESPCM, Evaluate, Spectra, GetModels
from Solution.DESPCM.Training import Compile, Train

# from Dataset.Simulation.GaussCurve_TF import FBG_spectra, GaussCurve
from Dataset.loader import DATASET_5fbg_1 as Dataset

import tensorflow as tf

print('loading dataset')
# [:,:,948-76:1269-76]
dataset = tf.constant(Dataset(), dtype=tf.dtypes.float32)
print('dataset load done')

# plt.plot(*dataset[12])
# plt.show()

I = [5.685, 2.919, 2.25, 0.9342, 0.4047]
W = tf.constant([0.1925, 0.1925, 0.1975, 0.1975, 0.2])*1.1
ITERATION = 300

de = DESPCM()
SpectraModel, SCM = GetModels()


def init(de):
    de.minx = 1545.0
    de.maxx = 1549.0
    de.I = I
    de.W = W
    de.NP = 100
    de.CR = 0.5
    de.F = 1


init(de)


def plot(data, X):

    simulation = Spectra(data[0], X, I, W)

    plt.plot(data[0], tf.transpose(simulation[0]), c='gray')
    plt.plot(data[0], tf.transpose(simulation[1]), c='green')
    plt.plot(*data, c='red')

    # A = simulation[0]
    # B = data[1]
    # plt.twinx()
    # plt.plot(data[0], (A-B)**2, c='green')
    # plt.twinx()
    # plt.plot(data[0], ((A-B)**2)/(A+B), c='orange')


def plotPause(info):

    i, data, X, V, dx, dv = info

    if i % 20 == 0:
        plt.clf()
        plot(data, X)
        plt.pause(0.02)

data = dataset[0]

# X = de.run(data, iterations=ITERATION, forEach=plotPause)
# X = tf.reduce_mean(X, axis=0)


Compile(SCM)

# for i in range(20):
#     Train(SCM, tf.expand_dims(data, axis=0), tf.expand_dims(X, axis=0), I, W)
#     plt.clf()
#     plot(data, tf.expand_dims(X, axis=0))
#     # plt.show()
#     plt.pause(0.02)

# plt.show()


X_log = []


def evaluateData(data):
     # data = dataset[10]
    print('run ------------------')
    # print(data)
    X = de(data, iterations=ITERATION)

    # X_log.append(X[tf.argmin(Evaluate(data, X, I, W))])
    X_log.append(tf.reduce_mean(X, axis=0))
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



for i in range(10):
    Xs = tf.map_fn(evaluateData, dataset)

    # for i in range(len(I))[::-1]:
    #     plt.fill_between(range(len(X_log)), np.array(
    #         min_log)[:, i], np.array(max_log)[:, i], alpha=0.3)

    print(Xs)
    plt.show()

    for i in range(20):
        Train(SCM, dataset, tf.reduce_mean(Xs, axis=1), I, W)
    de.compensate = 1

    X = de.run(data, iterations=ITERATION, forEach=plotPause)
    X = tf.reduce_mean(X, axis=0)
    plt.show()
