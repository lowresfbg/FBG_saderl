import matplotlib.pyplot as plt


from Solution.DE_general import DE, spectra_diff_contrast

from Dataset.Simulation.GaussCurve_TF import FBG_spectra, GaussCurve
from Dataset.loader import DATASET_3fbg_1 as Dataset

from Algorithms.PeakFinder import FindPeaks

from Tests.GeneralTest_logger import SingleLogger
import tensorflow as tf

print('loading dataset')
# [:,:,948-76:1269-76]
dataset = tf.constant(Dataset(), dtype=tf.dtypes.float32)

peaks = tf.constant(FindPeaks(dataset[0], 1e-5), dtype=tf.dtypes.float32)

I = peaks[:, 2]*1e3 
W = peaks[:, 0]*0.9

print('loading completed')
ITERATION = 1000

de = DE()


def init(de):
    de.minx = 1545.0
    de.maxx = 1549.0
    de.I = I
    de.W = W
    de.NP = 30
    de.CR = 0.7
    de.F = 1
    de.EarlyStop_threshold = 5e-3
    de.spectra_diff = spectra_diff_contrast


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

    plt.plot(X_log, "o")
    plt.twinx()
    plt.plot(err_log)

    plt.pause(0.01)
    return X


print("start multiple")
Xs = tf.map_fn(evaluateData, dataset)

# print(Xs)
plt.show()
