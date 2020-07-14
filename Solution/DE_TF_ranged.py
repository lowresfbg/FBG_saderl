import numpy as np
import matplotlib.pyplot as plt

from Algorithms.DifferentialEvolution import DifferentialEvolution_TF as de_alg
from Dataset.Simulation.GaussCurve_TF import FBG_spectra

import tensorflow as tf


@tf.function
def Evaluate(data, X, I, W):
    I = tf.repeat([I], X.shape[0], axis=0)
    # I = tf.concat([I[:, :1] + (X[:, :1] - 1546.52)
    #                * -0.35, I[:, 1:]], axis=1)
    W = tf.repeat([W], X.shape[0], axis=0)
    simulation = FBG_spectra(data[0], X, I, W)
    return spectra_diff(simulation, data[1])


@tf.function
def spectra_diff(A, B):
    return tf.reduce_sum((A-B)**2/(A+B), axis=1)


@tf.function
def loop(i, full_data, iterations, X, CR, F, max_xn, min_xn, I, W):
    # print(i)
    step = tf.maximum(1, tf.cast((1-i/iterations)*100, tf.dtypes.int32))
    data = full_data[:, ::step]
    # data = full_data

    V = de_alg.Mutate(X, CR, F)

    V = (V-min_xn)%(max_xn-min_xn)+min_xn

    dv = Evaluate(data, V, I,  W)
    dx = Evaluate(data, X,  I, W)

    X = X + (V-X) * \
        tf.expand_dims(tf.cast(dv < dx, tf.dtypes.float32), 1)

    return X, V, dx, dv


@tf.function
def computeRange(data, I):
    xs = tf.repeat([data[0][::10]], tf.shape(I)[0], axis=0)
    ys = tf.repeat([data[1][::10]], tf.shape(I)[0], axis=0)
    Is = tf.expand_dims(I, axis=1)*0.001*0.5

    mask = tf.cast(ys > Is, tf.dtypes.float32)

    max_xn = tf.reduce_max(xs*mask, axis=1)
    min_xn = tf.reduce_min(xs+(1-mask)*1e20, axis=1)
    # print(xs, ys,max_xn, min_xn)

    return max_xn+0.1, min_xn-0.1


class DE(tf.keras.Model):
    def __init__(self):
        super(DE, self).__init__()
        self.minx = 1545
        self.maxx = 1549
        self.I = []
        self.W = []

        self.NP = 10
        self.CR = 0.5
        self.F = 1.0

        self.iter = 0
        self.X = []
        self.V = []

    def loop_py(self, i, data, iterations, max_xn, min_xn, forEach):
        self.iter = i

        self.X, V, dx, dv = loop(tf.constant(i), data, iterations, self.X,
                                 self.CR,  self.F,
                                 max_xn, min_xn,
                                 self.I,  tf.constant(self.W))
        
        forEach([i, data, self.X, V, dx, dv])

        return (i+1,)

    def run(self, data, iterations=300, forEach=lambda x: x):
        max_xn, min_xn = computeRange(data, self.I)


        self.X = tf.random.uniform([self.NP, len(self.I)]) 
        max_xn = tf.repeat([max_xn], tf.shape(self.X)[0], axis=0)
        min_xn = tf.repeat([min_xn], tf.shape(self.X)[0], axis=0)

        self.X = self.X*(max_xn-min_xn)+min_xn

        i = 0
        self.iter = 0
        iterations = tf.constant(iterations)

        tf.while_loop(lambda _: self.iter >=0 and self.iter < iterations,
                      lambda i: self.loop_py(i, data, iterations, max_xn, min_xn, forEach), (i,))

        return self.X

    def __call__(self, inputs, **kwargs):
        return self.run(inputs, **kwargs)
