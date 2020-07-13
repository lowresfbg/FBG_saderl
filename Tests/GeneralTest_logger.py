import matplotlib.pyplot as plt
from Dataset.Simulation.GaussCurve_TF import FBG_spectra, GaussCurve
import tensorflow as tf

from Algorithms.PeakFinder import FindPeaks
import numpy as np

def plot(data, X, I, W):
    simulation = FBG_spectra(data[0], X, I, W)
    plt.plot(data[0], tf.transpose(simulation), c='gray')
    plt.plot(*data, c='red')

def formatNumber(value):
    return np.array(value).tolist()

class SingleLogger:
    def __init__(self, I, W):
        self.I = I
        self.W = W
        self.data = []
        self.X = []
        self.X_mean_log = []
        self.error_log = []
        self.iteration = []
        self.max_diff_log = []
        self.Animate = True

    def PlotFigure(self):
        plt.clf()
        plt.subplot(311)
        plot(self.data, self.X, self.I, self.W)
        plt.subplot(312)
        plt.plot(self.error_log)
        plt.yscale('log')
        plt.subplot(313)
        plt.plot(self.max_diff_log)
        plt.yscale('log')
        plt.title("iteration={}".format(self.iteration[-1]))
        plt.tight_layout()

    def GetData(self):
        return {
            "X": formatNumber(self.X),
            "err": formatNumber(self.error_log),
            "iteration": self.iteration[-1],
            "max_diff": formatNumber(self.max_diff_log),
            "data": formatNumber(self.data)
        }

    def log(self, info):
        i, data, X, V, dx, dv, max_dist = info
        self.X = X
        self.iteration.append(i)
        self.data = data
        self.X_mean_log.append(tf.reduce_mean(X, axis=0))
        self.error_log.append(tf.reduce_mean(tf.reduce_min([dx, dv], axis=0)))
        self.max_diff_log.append(max_dist)

        if self.Animate and i % 10 == 0:
            self.PlotFigure()

            plt.pause(0.02)


class DatasetConfig:
    def __init__(self, dataset):
        peaks = tf.constant(
            FindPeaks(dataset[0], 1e-5), dtype=tf.dtypes.float32)
        self.dataset = dataset
        self.I = peaks[:, 2]*1e3
        self.W = peaks[:, 0] #*0.9

    def Initiate(self, de):
        de.I = self.I
        de.W = self.W

    def GetData(self):
        return {
            "I": formatNumber(self.I),
            "W": formatNumber(self.W),
            # "dataset": formatNumber(self.dataset)
        }


class TestConfig:
    def __init__(self, NP, F, CR):
        self.NP = NP
        self.F = F
        self.CR = CR

    def Initiate(self, de):
        de.NP = self.NP
        de.F = self.F
        de.CR = self.CR

    def GetData(self):
        return {
            "NP": self.NP,
            "F": self.F,
            "CR": self.CR
        }


class CompleteTester:
    def __init__(self, name, dataset_config: DatasetConfig, config: TestConfig, de):
        self.name = name
        self.dataset_config = dataset_config
        self.config = config

        self.X_log = []
        self.max_iter = 1000

        self.DE = de
        self.Xs = []
        self.test_data = []

        # initiate DE
        self.config.Initiate(self.DE)
        self.dataset_config.Initiate(self.DE)

        self.DE.minx = 1545.0
        self.DE.maxx = 1549.0
        self.DE.EarlyStop_threshold = 1e-3

    def run(self):
        print('testing', self.name)
        self.Xs = tf.map_fn(self.evaluateData, self.dataset_config.dataset)

    def evaluateData(self, data):
        logger = SingleLogger(self.dataset_config.I, self.dataset_config.W)
        logger.Animate = False

        self.DE.afterEach.clear()
        self.DE.afterEach.append(logger.log)

        X = self.DE(data, max_iter=self.max_iter)
        self.X_log.append(tf.reduce_mean(X, axis=0))
        self.test_data.append(logger.GetData())
        return X

    def GetData(self):
        return {
            "config": self.config.GetData(),
            "test_data": self.test_data,
            "dataset": self.dataset_config.GetData(),
            "X_log": formatNumber(self.X_log),
            "Xs": formatNumber(self.Xs)
        }
