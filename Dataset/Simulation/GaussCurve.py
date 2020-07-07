import numpy as np
def GaussCurve(x, I, C, W):
    return I*np.exp(-np.sqrt(2)/8*((x-C)/W)**2)