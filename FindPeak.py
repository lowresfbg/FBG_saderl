import numpy as np
import csv

from Algorithms.PeakFinder import FindPeaks
from Dataset.loader import DATASET_3fbg_1 as Dataset
dataset = Dataset()

with open('peak.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['data','peak1','peak2','peak3'])
    for i in range(0,len(dataset)):
        A = np.array(FindPeaks(dataset[i], 1e-5)).astype(float)[:,1].T
        writer.writerow([i,*A])

    
