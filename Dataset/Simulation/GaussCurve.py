import numpy as np
from numba import jit

@jit
def GaussCurve(x, I, C, W):
    return I*np.exp(-np.sqrt(2)/8*((x-C)/W)**2)

@jit
def JIT_FBG_spectra(x_coord, X, I, W):
    X = np.expand_dims(X, axis=len(X.shape))
    return np.sum(GaussCurve(x_coord, I, X, W), axis=1)

def FBG_spectra(x_coord, X, I, W=0.05):
    x_coord = np.tile(x_coord, X.shape+(1,))
    I = np.expand_dims(np.array(I), axis=1)*0.001
    return JIT_FBG_spectra(x_coord, X, I, W)