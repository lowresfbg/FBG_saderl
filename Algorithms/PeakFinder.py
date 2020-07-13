from Algorithms import Find_FWHM
import matplotlib.pyplot as plt
import numpy as np


def FindPeaks(data, threshold=1e-5):
    data = np.array(data)

    mask = (data[1] > threshold)*1

    size = len(data[0])
    peakup = np.arange(size-1)[mask[1:] > mask[:-1]]
    peakdown = np.arange(size-1)[mask[1:] < mask[:-1]]

    peaks = []
    for i in range(len(peakup)):
        up = peakup[i]
        down = peakdown[i]
        FWHM, pos, I = Find_FWHM.findFWHM(data, up, down)
        peaks.append([FWHM, pos, I])
        # print(FWHM, pos, I)
    return peaks
    
