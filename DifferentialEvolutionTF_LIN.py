import numpy as np
import matplotlib.pyplot as plt
import time

from Solution.DE_TF import DE as DETF
from Solution.DE_TF import Evaluate

from Dataset.Simulation.GaussCurve_TF import FBG_spectra, GaussCurve
from Dataset.loader import DATASET_5fbg_1 as Dataset

import tensorflow as tf

print('loading dataset')
dataset = tf.constant(Dataset(), dtype=tf.dtypes.float32)#[:,:,948-76:1269-76]
print('dataset load done')

I = [5.685, 2.919, 2.25, 0.9342, 0.4047]
W = tf.constant([0.1925, 0.1925, 0.1975, 0.1975, 0.2])
ITERATION = 500

start =time.time()
detf = DETF()


def init(de):
    de.minx = 1545.0
    de.maxx = 1549.0
    de.I = I
    de.W = W
    de.NP = 50
    de.CR = 0.5
    de.F = 1


init(detf)

def DynamicAdjust(info):
    
    i, data, X, V, dx, dv = info
    if ((i-1) % 20 == 0) & (i-1>100):
        #detf.F = tf.constant(detf.F - 0.03)
        detf.CR = tf.constant(detf.CR - 0.0)
    print()
    return

class Plotter():
    def __init__(self):
        self.log = []

    def logger(self, info):
        i, data, X, V, dx, dv = info
        self.log.append(np.min([dx, dv]))

def plot(data, X):

    simulation = FBG_spectra(data[0], X, tf.repeat([I], X.shape[0], axis=0), W)

    plt.plot(data[0], tf.transpose(simulation), c='gray')
    plt.plot(*data, c='red')
    plt.plot(data[0],(data[1]-tf.transpose(simulation[0])),c='yellow')

    # A = simulation[0]
    # B = data[1]
    # plt.twinx()
    # plt.plot(data[0], (A-B)**2, c='green')
    # plt.twinx()
    # plt.plot(data[0], ((A-B)**2)/(A+B), c='orange')

plotter = Plotter()

def plotPause(info):
    DynamicAdjust(info)
    plotter.logger(info)
    i, data, X, V, dx, dv = info
    if i == ITERATION:
        plt.clf()
        plt.subplot(211)
        end = time.time()
        title= "done!and it takes " + str(end-start) + " seconds."
        plt.title(title)
        plot(data, X)
        
        plt.subplot(212)
        plt.plot(plotter.log)
        plt.yscale("log")

        plt.pause(0.00001)
        



detf.run(dataset[20], iterations=ITERATION, forEach=plotPause)
#detf.run(dataset[0], iterations=ITERATION, forEach=plotPause)
plt.show()





X_log = []
# max_log = []
# min_log = []






def evaluateData(data):
    init(detf)
    start = time.time()
     # data = dataset[10]
    print('run ------------------')
    # print(data)
    X = detf(data, iterations=ITERATION, forEach=DynamicAdjust)

    #Xmean = tf.reduce_mean(X, axis=0)
    #X_log.append(Xmean)
    X_log.append(X[tf.argmin(Evaluate(data, X, I, W))])
    end = time.time()
    title= "done!and it takes " + str(end - start) + " seconds."
    plt.clf()
    plt.title(title)

    plt.axhline(1546.923, linestyle=":")
    plt.axhline(1547.29875, linestyle=":")
    plt.axhline(1547.65375, linestyle=":")
    plt.axhline(1548.015, linestyle=":")

    # max_xn, min_xn = computeRange(data, I)

    # max_log.append(max_xn)
    # min_log.append(min_xn)
    plt.plot(X_log)
    plt.pause(0.00001)
    
    return X

#Xs = tf.map_fn(evaluateData, dataset)

# for i in range(len(I))[::-1]:
#     plt.fill_between(range(len(X_log)), np.array(
#         min_log)[:, i], np.array(max_log)[:, i], alpha=0.3)

#print(Xs)
plt.show()
