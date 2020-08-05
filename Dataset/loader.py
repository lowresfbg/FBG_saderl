import os
from . import reader
import re
import csv
import numpy as np

dirname = os.path.dirname(__file__)

# loading helper


def load_folder(folder):
    data = []
    answer = None
    folder_name = os.path.join(dirname, folder)

    for f in os.listdir(folder_name):
        if re.match('.*\\.csv$', f):
            filepath = os.path.join(folder_name, f)
            data.append(reader.read_csv(filepath))

        if re.match('answer.txt', f):
            filepath = os.path.join(folder_name, f)

            with open(filepath, 'r') as f:
                answer = np.array(list(csv.reader(f)))[:, 1:].astype(float)

    if answer is not None:
        data = (data, answer)
        
    return data

# load data sets


# 10 FBG
def DATASET_10fbg_1():
    return load_folder("Measured/10fbg/")

# 7 FBG
def DATASET_7fbg_1():
    return load_folder("Measured/7fbg/")
    
# 5 FBG


def DATASET_5fbg_1():
    return load_folder("Measured/5fbg/strain FBG1/")


def DATASET_5fbg_1_1():
    return load_folder("Measured/5fbg/strain FBG1/only FBG1")


def DATASET_5fbg_2():
    return load_folder("Measured/5fbg/strain FBG1 _ FBG2")


def DATASET_5fbg_2_1():
    return load_folder("Measured/5fbg/strain FBG1 _ FBG2/only FBG1(2 units)")


def DATASET_5fbg_2_2():
    return load_folder("Measured/5fbg/strain FBG1 _ FBG2/only FBG2")


def DATASET_5fbg_3():
    return load_folder("Measured/5fbg/strain FBG1 _ FBG2 _ FBG3")


def DATASET_5fbg_3_1():
    return load_folder("Measured/5fbg/strain FBG1 _ FBG2 _ FBG3/only FBG1(3 units)")


def DATASET_5fbg_3_perfect():
    return load_folder("Measured/5fbg/perfect_3move")

def DATASET_5fbg_3_2():
    return load_folder("Measured/5fbg/strain FBG1 _ FBG2 _ FBG3/only FBG2(2 units)")


def DATASET_5fbg_3_3():
    return load_folder("Measured/5fbg/strain FBG1 _ FBG2 _ FBG3/only FBG3")


def DATASET_5fbg_1_perfect():
    return load_folder("Measured/5fbg/perfect5")

# 3 FBG


def DATASET_3fbg_1():
    return load_folder("Measured/3fbg/strain FBG1")


def DATASET_3fbg_1_2():
    return load_folder("Measured/3fbg/strain FBG1-2")


def DATASET_3fbg_perfect():
    return load_folder("Measured/3fbg/perfect")


def DATASET_3fbg_2():
    return load_folder("Measured/3fbg/2move")


def DATASET_3fbg_2_noise():
    return load_folder("Measured/3fbg/2noisemove")


# 1 FBG

def DATASET_1fbg_differentResolution():
    return load_folder("Measured/1fbg/DifferentResolution")

# background

def DATASET_background():
    return load_folder("Measured/background")