import matplotlib.pyplot as plt
import numpy as np
from Algorithms.PeakFinder import FindPeaks
from Dataset.loader import DATASET_3fbg_1 as Dataset
import csv


def Get3FBGAnswer():
    dataset = Dataset()
    ids = []
    wls = []

    fbg_count = 3
    threshold = 1e-5

    for i, data in enumerate(dataset):
        peaks = np.array(FindPeaks(data, threshold))
        if len(peaks) == fbg_count:
            ids.append(i)
            wls.append(peaks[np.argsort(peaks[:, 2])][::-1, 1])

    ids = np.array(ids)
    wls = np.array(wls)

    ids_c = np.arange(min(ids), max(ids)+1)
    data = []
    for i in range(fbg_count):
        wl = wls[:, i]
        w, b = np.polyfit(ids,wl,1)
        # plt.plot(ids, wl, "o-")
        # plt.plot(ids_c, ids_c*w+b, "o-")
        data.append(ids_c*w+b)

    # plt.plot(ids, wls, "o-")
    # plt.show()
    # print(peaks)
    return data


def GetFBGAnswer(dataset, fbg_count, threshold):
    ids = []
    wls = []


    for i, data in enumerate(dataset):
        peaks = np.array(FindPeaks(data, threshold))
        if len(peaks) == fbg_count:
            ids.append(i)
            wls.append(peaks[np.argsort(peaks[:, 2])][::-1, 1])

    ids = np.array(ids)
    wls = np.array(wls)

    ids_c = np.arange(min(ids), max(ids)+1)
    data = []
    for i in range(fbg_count):
        wl = wls[:, i]
        w, b = np.polyfit(ids,wl,1)
        # plt.plot(ids, wl, "o-")
        # plt.plot(ids_c, ids_c*w+b, "o-")
        data.append(ids_c*w+b)

    # plt.plot(ids, wls, "o-")
    # plt.show()
    # print(peaks)
    return data