import csv
import numpy as np

def read_csv(path) -> np.ndarray:
    """Read OSA csv file and return spectrum data

    Args:
        path (str): the path to the file

    Returns:
        np.array: in shape(0=wavelength/1=intensity, samples)
    """
    
    with open(path) as f:
        data = np.array(list(csv.reader(f))[75:])
        return data[:,:2].astype(float).T

if __name__ == "__main__":
    print(read_csv('./Dataset/5fbg/strain FBG1/WaveData20200304_250.csv'))