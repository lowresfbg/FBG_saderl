import json
import os
import re

dirname = os.path.dirname(__file__)


def read(filepath):
    with open(filepath, 'r') as f:
        obj = json.load(f)
    return obj

# loading helper
def load_folder(folder):
    data = []
    folder_name = os.path.join(dirname, folder)
    for f in os.listdir(folder_name):
        if re.match('.*\\.json$', f):
            filepath = os.path.join(folder_name, f)
            data.append(read(filepath))
    return data


def ResultSet_EarlyStop():
    return load_folder('EarlyStop')


def ResultSet_Iter200():
    return load_folder('Iter200')


def ResultSet_CR():
    return load_folder('CR_EarlyStop')

def ResultSet_FWHM():
    return load_folder('FWHM_mul')