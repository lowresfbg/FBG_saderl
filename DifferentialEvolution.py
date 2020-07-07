import numpy as np
import matplotlib.pyplot as plt

from Algorithms.DifferentialEvolution import DE

from Dataset.Simulation.GaussCurve import GaussCurve

from Dataset.loader import DATASET_5fbg_2

dataset = DATASET_5fbg_2()

minx = 1545
maxx = 1549



def FBG(x_coord, X):
    x_coord = np.tile(x_coord, X.shape+(1,))
    X = np.expand_dims(X, axis=len(X.shape))
    I = np.array([5.72, 2.95, 2.2, 1, 0.5])[:, np.newaxis]*0.001
    return np.sum(GaussCurve(x_coord, I, X, 0.05), axis=1)


def difference(data, X):
    simulation = FBG(data[0], X)
    return np.mean(((simulation-data[1])/(simulation+data[1]+0.006))**2, axis=1)


def run(data):
    X = np.random.rand(15,5)*(maxx-minx)+minx


    for i in range(300):
        
        V = DE.Mutate(X, 0.5, 1)

        V = np.clip(V, minx, maxx)

        X = X + (V-X)*(difference(data,V)<difference(data,X))[:,np.newaxis]

        MSE = min(difference(data,X))


        if i%10==0:

            plt.clf()
            plt.plot(data[0], FBG(data[0],X).T, c='gray')
            plt.plot(*data, c='red')
            plt.title("Iter={}, MSE={}".format(i, MSE))
            plt.pause(0.01)
    
    plt.show()
    return X

run(dataset[10])

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