import numpy as np
import matplotlib.pyplot as plt

from Algorithms.DifferentialEvolution import DE
from Dataset.Simulation.GaussCurve import FBG_spectra


def spectra_diff(data, X):
    simulation = FBG_spectra(data[0], X)
    return np.mean((simulation-data[1])**2, axis=1)

class DifferentialEvolution():
    def __init__(self):
        self.minx = 1545
        self.maxx = 1549

    def run(self,data, iterations=300, forEach=lambda x: x):
        X = np.random.rand(15, 5)*(self.maxx-self.minx)+self.minx

        for i in range(iterations):

            V = DE.Mutate(X, 0.5, 1)

            V = np.clip(V, self.minx, self.maxx)

            dv = spectra_diff(data, V)
            dx = spectra_diff(data, X)

            X = X + (V-X)*(dv < dx)[:, np.newaxis]

            forEach([i,data, X, V, dx, dv])

        return X


