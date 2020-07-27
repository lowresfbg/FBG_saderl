import numpy as np
from Algorithms.PeakFinder import FindPeaks


def GetFBGAnswer(dataset, fbg_count, threshold):
    ids = []
    W = []
    C = []
    I = []

    for i, data in enumerate(dataset):
        peaks = np.array(FindPeaks(data, threshold))
        if len(peaks) == fbg_count:
            ids.append(i)
            W.append(peaks[np.argsort(peaks[:, 2])][::-1, 0])
            C.append(peaks[np.argsort(peaks[:, 2])][::-1, 1])
            I.append(peaks[np.argsort(peaks[:, 2])][::-1, 2])

    ids = np.array(ids)
    W = np.array(W)
    C = np.array(C)
    I = np.array(I)

    ids_c = np.arange(min(ids), max(ids)+1)
    data = []
    
    for i in range(fbg_count):
        c = C[:, i]
        w, b = np.polyfit(ids,c,1)
        # plt.plot(ids, wl, "o-")
        # plt.plot(ids_c, ids_c*w+b, "o-")
        data.append(ids_c*w+b)

    # plt.plot(ids, wls, "o-")
    # plt.show()
    # print(peaks)
    return data, np.average(I), np.average(W)