import matplotlib.pyplot as plt
from cycler import cycler
default_cycler = (cycler(color=[
    '#3f51b5',
    '#ff5722',
    '#4caf50',
    '#e91e63',
    '#9c27b0',
    '#2196f3',
    '#fbc02d']))
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = default_cycler


from matplotlib.ticker import MaxNLocator

#...

# ---------------

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from Solution.DE_general import DE, spectra_diff_contrast, spectra_diff_absolute

from Dataset.Simulation.GaussCurve_TF import FBG_spectra, GaussCurve

# DATASET!!!!!!!!!!!!!
from Dataset.loader import DATASET_3fbg_1_2 as Dataset
# from Dataset.loader import DATASET_7fbg_1 as Dataset
from Dataset.loader import DATASET_background
from Dataset import Resampler

from Algorithms.PeakFinder import FindPeaks

from Tests.GeneralTest_logger import SingleLogger
import tensorflow as tf

import numpy as np
from AutoFitAnswer import Get3FBGAnswer


print('loading dataset')
# [:,:,948-76:1269-76]

# dataset = Dataset()[0]
dataset = Dataset()
fbgs=3



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


# dataset, answer, peaks = Resampler.Resample((newDataset, Dataset()[1]), fbgs, 1, 1000, threshold)
dataset, answer, peaks = Resampler.Resample(newDataset, fbgs, 1, 1000, threshold)

print(dataset.shape)


print(peaks)


I = peaks[:, 2] *1e3
W = peaks[:, 0] #*0.9


print('loading completed')
ITERATION = 3000

de = DE()


def init(de):
    de.minx = 1544.0
    de.maxx = 1550.0
    de.I = I
    de.W = W
    de.NP = 200
    de.CR = 0.8
    de.F = 0.5
    de.EarlyStop_threshold = 1e-3
    # de.spectra_diff = spectra_diff_absolute


init(de)


sl = SingleLogger(I, W)

# RUN SINGLE TEST
# print("start single test")

# de.afterEach.append(sl.log)
# de.run(dataset[3], max_iter=ITERATION)
# sl.PlotFigure()
# plt.show()
# print("single test complete")


X_log = []
err_log = []

de.afterEach.clear()

plt.figure(figsize=(3.89,3.98),dpi=150)

def evaluateData(data):
    print('run ------------------')
    X = de(data, max_iter=ITERATION)

    X_log.append(tf.reduce_mean(X, axis=0))

    plt.clf()

    plt.plot(answer,"--" ,color='gray')
    for i in range(X_log[0].shape[0]):
        plt.plot(np.array(X_log)[:,i], "o", label="FBG{}".format(i+1))
    

    # plt.twinx()
    # plt.plot(err_log)
    plt.legend(fontsize=6)
    # plt.legend(fontsize=6, ncol=2)
    plt.xlabel("Tests\n"+ r"$\bf{(b)}$")
    plt.ylabel("Wavelength (nm)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.pause(0.01)
    return X



print("start multiple")
Xs = tf.map_fn(evaluateData, dataset)

X_log = np.array(X_log)
error = np.sqrt(np.mean((X_log - answer)**2))
print(error)
# print(Xs)
plt.show()
