import numpy as np
import matplotlib.pyplot as plt

from Solution.DE_newdiff import DE_newdiff
from Solution.DE import DE

from Dataset.Simulation.GaussCurve import FBG_spectra


from Dataset.Simulation.GaussCurve import GaussCurve

from Dataset.loader import DATASET_5fbg_1 as Dataset

print('loading dataset')
dataset = Dataset()
print('dataset load done')

I = [5.72, 2.95, 2.2, 1, 0.5]
W = 0.04

de = DE()
de_newdiff = DE_newdiff()

de.minx = 1545
de.maxx = 1549
de.I = I
de.W = W
de.NP = 40

de_newdiff.minx = 1545
de_newdiff.maxx = 1549
de_newdiff.I = I
de_newdiff.W = W
de_newdiff.NP = 50


def plot(data, X):

    simulation = FBG_spectra(data[0], X, I)

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

    if i%10==0:
        plt.clf()
        plot(data, X)
        plt.pause(0.02)


# de_newdiff.run(dataset[26], iterations=400, forEach=plotPause)
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
