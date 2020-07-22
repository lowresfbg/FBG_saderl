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


print('loading dataset')
# [:,:,948-76:1269-76]

dataset = Dataset()

# HERE IS THE MAGIC...
background = Resampler.Sample(DATASET_background(), len(dataset[0][0]),50)[0][1]

newDataset = []
for data in dataset:
    newDataset.append(np.array([
            data[0],
            data[1] / background
        ]))

ymax = np.max(newDataset[0][1])
ymin = np.min(newDataset[0][1])
threshold = (ymax-ymin)*0.1+ymin


dataset, answer, peaks = Resampler.Resample(newDataset, 3, 1, 1000, threshold)


# feature = spectra_feature(dataset[:,1], tf.reduce_mean(dataset[:,1]), tf.reduce_max(dataset[:,1]), 100)
# print(feature)
# plt.plot(tf.transpose(feature))
# plt.show()

# plt.plot(*background[0])
# plt.show()

# dataset = tf.constant(Dataset(), dtype=tf.dtypes.float32)[::4,:,::5]
# answer = tf.constant(np.array(Get3FBGAnswer()).T, dtype=tf.dtypes.float32)[::4]
print(dataset.shape)


# from scipy.interpolate import interp1d

# plt.plot(*dataset[5])

# x_coord = np.linspace(np.min(dataset[:,0]), np.max(dataset[:,0]), 10000)
# new_dataset = []
# for i in range(dataset.shape[0]):
#     new_dataset.append(interp1d(dataset[i,0], dataset[i,1],  kind='cubic')(x_coord))
# dataset = tf.cast(tf.concat([
#     tf.repeat([[x_coord]], dataset.shape[0], axis=0),
#     np.expand_dims(new_dataset, axis=1)], axis=1), tf.dtypes.float32)

# print(dataset.shape)

# plt.plot(*dataset[5])
# plt.show()

# peaks = tf.constant(FindPeaks(dataset[0], 1e-5), dtype=tf.dtypes.float32)
# peaks = tf.constant(FindPeaks(dataset[17], 0.12), dtype=tf.dtypes.float32)

print(peaks)


I = peaks[:, 2]*1e3
W = peaks[:, 0]*0.9

# for perfect
# I = tf.constant([1, 0.5, 0.25], dtype=tf.dtypes.float32)*1e3 
# W = tf.constant([0.5264,0.5264,0.5264], dtype=tf.dtypes.float32)

# ## answer
# spectra = FBG_spectra(dataset[0][0], answer, I, W)
# dataset = tf.concat(
#     [
#         tf.expand_dims(dataset[:,0], axis=1),
#         tf.expand_dims(spectra+(tf.random.uniform(spectra.shape)-0.5)*1e-5,axis=1)
#     ],
#     axis=1)

# print(dataset)
# 


print('loading completed')
ITERATION = 500

de = DE()


def init(de):
    de.minx = 1545.0
    de.maxx = 1549.0
    de.I = I
    de.W = W
    de.NP = 200
    de.CR = 0.6
    de.F = 0.99
    de.EarlyStop_threshold = 1e-3
    de.spectra_diff = spectra_diff_absolute


init(de)


sl = SingleLogger(I, W)

# RUN SINGLE TEST
print("start single test")

de.afterEach.append(sl.log)
de.run(dataset[3], max_iter=ITERATION)
sl.PlotFigure()
plt.show()
print("single test complete")


X_log = []
err_log = []

de.afterEach.clear()


def evaluateData(data):
    print('run ------------------')
    X = de(data, max_iter=ITERATION)

    X_log.append(tf.reduce_mean(X, axis=0))

    plt.clf()

    plt.plot(answer,"--" ,color='gray')
    for i in range(X_log[0].shape[0]):
        plt.plot(np.array(X_log)[:,i], "o-", label="FBG{}".format(i+1))
    

    # plt.twinx()
    # plt.plot(err_log)
    plt.legend()
    plt.xlabel("Test set")
    plt.ylabel("Wavelength")
    plt.pause(0.01)
    plt.tight_layout()
    return X



print("start multiple")
Xs = tf.map_fn(evaluateData, dataset)

X_log = np.array(X_log)
error = np.sqrt(np.mean((X_log - answer)**2))
print(error)
# print(Xs)
plt.show()
