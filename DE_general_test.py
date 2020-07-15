import matplotlib.pyplot as plt


from Solution.DE_general import DE, spectra_diff_contrast, spectra_diff_absolute

from Dataset.Simulation.GaussCurve_TF import FBG_spectra, GaussCurve

# DATASET!!!!!!!!!!!!!
from Dataset.loader import DATASET_3fbg_2_noise as Dataset

from Algorithms.PeakFinder import FindPeaks

from Tests.GeneralTest_logger import SingleLogger
import tensorflow as tf

import numpy as np
from AutoFitAnswer import Get3FBGAnswer


print('loading dataset')
# [:,:,948-76:1269-76]
dataset = tf.constant(Dataset(), dtype=tf.dtypes.float32)

# plt.plot(*dataset[17])
# plt.show()

# peaks = tf.constant(FindPeaks(dataset[0], 1e-5), dtype=tf.dtypes.float32)
peaks = tf.constant(FindPeaks(dataset[17], 0.12), dtype=tf.dtypes.float32)

print(peaks)


# I = peaks[:, 2]*1e3 
# W = peaks[:, 0]*0.9

I = tf.constant([1, 0.5, 0.25], dtype=tf.dtypes.float32)*1e3 
W = tf.constant([0.5264,0.5264,0.5264], dtype=tf.dtypes.float32)

# ## answer
# answer = tf.constant(np.array(Get3FBGAnswer()).T, dtype=tf.dtypes.float32)
# spectra = FBG_spectra(dataset[0][0], answer, I, W)
# dataset = tf.concat(
#     [
#         tf.expand_dims(dataset[:,0], axis=1),
#         tf.expand_dims(spectra+(tf.random.uniform(spectra.shape)-0.5)*1e-5,axis=1)
#     ],
#     axis=1)

print(dataset)



print('loading completed')
ITERATION = 1000

de = DE()


def init(de):
    de.minx = 1545.0
    de.maxx = 1549.0
    de.I = I
    de.W = W
    de.NP = 50
    de.CR = 0.6
    de.F = 1
    de.EarlyStop_threshold = 1e-2
    de.spectra_diff = spectra_diff_absolute


init(de)


sl = SingleLogger(I, W)

print("start single test")

de.afterEach.append(sl.log)
de.run(dataset[6], max_iter=ITERATION)
print("single test complete")
sl.PlotFigure()
plt.show()


X_log = []
err_log = []

de.afterEach.clear()


def evaluateData(data):
    print('run ------------------')
    X = de(data, max_iter=ITERATION)

    X_log.append(tf.reduce_mean(X, axis=0))

    plt.clf()

    plt.plot(X_log, "o-")
    plt.twinx()
    plt.plot(err_log)

    plt.pause(0.01)
    return X


print("start multiple")
Xs = tf.map_fn(evaluateData, dataset)

# print(Xs)
plt.show()
