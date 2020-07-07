import numpy as np
import matplotlib.pyplot as plt

from Solution.BasicDifferentialEvolution import DifferentialEvolution
from Dataset.Simulation.GaussCurve import FBG_spectra

from Algorithms.DifferentialEvolution import DE

from Dataset.Simulation.GaussCurve import GaussCurve

from Dataset.loader import DATASET_5fbg_2

print('loading dataset')
dataset = DATASET_5fbg_2()
print('dataset load done')

DE = DifferentialEvolution()
DE.minx = 1545
DE.maxx = 1549



def plot(info):
    i, data, X, V, dx, dv = info

    MSE = np.min([dx, dv])

    plt.clf()
    plt.plot(data[0], FBG_spectra(data[0], X).T, c='gray')
    plt.plot(*data, c='red')
    plt.title("Iter={}, MSE={}".format(i, MSE))
    plt.pause(0.01)


DE.run(dataset[10], forEach=plot)

# X_log = []

# for idata, data in enumerate(dataset):
#     # data = dataset[10]
#     print(idata,'/', len(dataset))
#     X = run(data)


#     X_log.append(X[np.argmin(difference(data,X))])
#     plt.clf()
#     plt.plot(X_log)
#     plt.pause(0.01)


# plt.show()
