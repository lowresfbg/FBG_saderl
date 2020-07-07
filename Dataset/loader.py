import os
from . import reader
import re

dirname = os.path.dirname(__file__)

def load_folder(folder):
    data = []
    folder_name = os.path.join(dirname, folder)
    for f in os.listdir(folder_name):
        if re.match('.*\\.csv$', f):
            filepath = os.path.join(folder_name, f)
            data.append(reader.read_csv(filepath))
    return data


def DATASET_5fbg_1():
    return load_folder("Measured/5fbg/strain FBG1")

def DATASET_5fbg_2():
    return load_folder("Measured/5fbg/strain FBG1 _ FBG2")





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    import numpy as np
    data = DATASET_5fbg_1()

    for i,d in enumerate(data):
        ax.plot(*d, zs=i, zdir='y')

    plt.show()

    