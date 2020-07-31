import matplotlib.pyplot as plt

from Solution.DE_general import DE, spectra_diff_contrast, spectra_diff_absolute

from Dataset.Simulation.GaussCurve_TF import FBG_spectra, GaussCurve

# DATASET!!!!!!!!!!!!!
from Dataset.loader import DATASET_3fbg_1_2 as Dataset
from Dataset.loader import DATASET_background
from Dataset import Resampler

from Algorithms.PeakFinder import FindPeaks

from Tests.GeneralTest_logger import SingleLogger
import tensorflow as tf

import numpy as np
from AutoFitAnswer import Get3FBGAnswer


import matplotlib

cmap = matplotlib.cm.get_cmap('Spectral')


print('loading dataset')
# [:,:,948-76:1269-76]

fbgs = 5
samples = 40




def normalize(spectra):
    maximum = tf.expand_dims(tf.reduce_max(spectra, axis=1), axis=1)
    minimum = tf.expand_dims(tf.reduce_min(spectra, axis=1), axis=1)
    return (spectra-minimum)/(maximum-minimum)


x_coord = tf.linspace(1545.0, 1548.0, 1000)

X1 = tf.random.uniform([samples, fbgs])*(1548-1545)+1545
I1 = tf.repeat([[1,0.6,0.3,0.2,0.1]], samples, axis=0)*1e3 + tf.random.uniform([samples, fbgs])
W1 = tf.ones([samples, fbgs]) * 0.04
spectrums1 = FBG_spectra(x_coord, X1, I1, W1)



print('loading completed')
ITERATION = 1000

de = DE()

def giveAnswer(info, answer):
    amount = 10
    de.X = tf.concat([ tf.repeat([answer],amount, axis=0), de.X[amount:]], axis=0)



def giveAnswerGenerator(answer):
    return lambda info : giveAnswer(info, answer)

def init(de, answer):
    de.minx = 1545.0
    de.maxx = 1549.0
    de.I = I1[0]
    de.W = W1[0]
    de.NP = 50
    de.CR = 0.8
    de.F = 0.5
    de.EarlyStop_threshold = 1e-2
    de.spectra_diff = spectra_diff_absolute
    # de.beforeEach.append(giveAnswerGenerator(answer))




sl = SingleLogger(I1, W1)

# print("start single test")

# de.afterEach.append(sl.log)
# de.run(dataset[3], max_iter=ITERATION)
# sl.PlotFigure()
# plt.show()
# print("single test complete")


X_log = []
err_log = []

de.afterEach.clear()
print(x_coord)
print(spectrums1[0])

data = tf.concat([
    [x_coord],
    [spectrums1[0]]
], axis=0)
print(data.shape)


input_error_mse_log = []
iterations_log = []
output_error_mse_log = []
c = []



for rt in range(100):
    for i in range(spectrums1.shape[0]):
        input_error = (tf.random.uniform([fbgs])-0.5)*2
        input_error_mse = tf.reduce_mean(input_error**2)

        data = tf.concat([
            [x_coord],
            [spectrums1[i]]
        ], axis=0)

        print('run ------------------')

        answer = X1[i]+input_error
        init(de, answer )

        X = de(data, max_iter=ITERATION)

        x_mean = tf.reduce_mean(X, axis=0)

        iterations_log.append(de.iter)
        output_error_mse_log.append(tf.reduce_mean((x_mean-answer)**2))

        c.append(cmap(input_error_mse))

        plt.clf()

        plt.scatter(np.array(output_error_mse_log), np.array(iterations_log), c=c)


        plt.pause(0.01)
        plt.tight_layout()


plt.show()