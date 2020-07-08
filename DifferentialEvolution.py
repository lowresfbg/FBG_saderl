import numpy as np
import matplotlib.pyplot as plt

from Solution.DE_newdiff import DE_newdiff
from Solution.DE import DE

from Dataset.Simulation.GaussCurve import FBG_spectra, GaussCurve



from Dataset.loader import DATASET_5fbg_1 as Dataset

print('loading dataset')
dataset = Dataset()
print('dataset load done')

I = [5.685, 2.919, 2.25, 0.9342, 0.4047]
W = 0.2

de = DE()
de_newdiff = DE_newdiff()


def init(de):
    de.minx = 1545
    de.maxx = 1549
    de.I = I
    de.W = W
    de.NP = 50


init(de)
init(de_newdiff)


def plot(data, X):

    simulation = FBG_spectra(data[0], X, np.repeat([I], X.shape[0], axis=0), W)

    plt.plot(data[0], simulation.T, c='gray')
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


# de_newdiff.run(dataset[20], iterations=1000, forEach=plotPause)
# plt.show()


class Plotter():
    def __init__(self):
        self.log = []

    def logger(self, info):
        i, data, X, V, dx, dv = info
        self.log.append(np.min([dx, dv]))


def compare():
    p1 = Plotter()
    p2 = Plotter()

    data = dataset[40]
    X1 = de.run(data, iterations=200, forEach=p1.logger)
    X2 = de_newdiff.run(data, iterations=200, forEach=p2.logger)

    print(p1.log)
    print(p2.log)

    plt.subplot(221)
    plt.plot(p1.log)

    plt.subplot(223)
    plt.plot(p2.log)

    plt.subplot(222)
    plot(data, X1)

    plt.subplot(224)
    plot(data, X2)

    plt.show()


X_log = []



for idata, data in enumerate(dataset):
    # data = dataset[10]
    print(idata, '/', len(dataset))
    X = de_newdiff.run(data, iterations=500)

    X_log.append(X[np.argmin(de.Evaluate(data, X))])
    plt.clf()
    plt.plot(X_log)
    plt.pause(0.01)


plt.show()
