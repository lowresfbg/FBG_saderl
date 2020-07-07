import os
from . import reader
import re

dirname = os.path.dirname(__file__)

# loading helper
def load_folder(folder):
    data = []
    folder_name = os.path.join(dirname, folder)
    for f in os.listdir(folder_name):
        if re.match('.*\\.csv$', f):
            filepath = os.path.join(folder_name, f)
            data.append(reader.read_csv(filepath))
    return data

# load data sets
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

def DATASET_5fbg_3_2():
    return load_folder("Measured/5fbg/strain FBG1 _ FBG2 _ FBG3/only FBG2(2 units)")

def DATASET_5fbg_3_3():
    return load_folder("Measured/5fbg/strain FBG1 _ FBG2 _ FBG3/only FBG3")

    