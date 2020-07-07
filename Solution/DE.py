import numpy as np
import matplotlib.pyplot as plt

from Algorithms.DifferentialEvolution import DifferentialEvolution as de_alg
from Dataset.Simulation.GaussCurve import FBG_spectra


from numba import jit

@jit
def JIT_spectra_diff(A, B):
    return np.sum((A-B)**2, axis=1)


class DE():
    def __init__(self):
        self.minx = 1545
        self.maxx = 1549
        self.I = []
        self.W = 0.04

        self.NP = 10
        self.CR = 0.5
        self.F = 1.5

    def Evaluate(self, data, X):
        simulation = FBG_spectra(data[0], X, self.I, self.W)
        return self.spectra_diff(simulation, data[1])

    def spectra_diff(self, A, B):
        return JIT_spectra_diff(A,B)

    def run(self, data, iterations=300, forEach=lambda x: x):
        X = np.random.rand(self.NP, 5)*(self.maxx-self.minx)+self.minx

        for i in range(iterations):

            V = de_alg.Mutate(X, self.CR, self.F)

            V = np.clip(V, self.minx, self.maxx)

            dv = self.Evaluate(data, V)
            dx = self.Evaluate(data, X)

            X = X + (V-X)*(dv < dx)[:, np.newaxis]

            forEach([i, data, X, V, dx, dv])

        return X
