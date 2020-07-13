from Algorithms.PeakFinder import FindPeaks
from Dataset.loader import DATASET_3fbg_1 as Dataset
dataset = Dataset()
peaks = FindPeaks(dataset[0], 1e-5)
print(peaks)