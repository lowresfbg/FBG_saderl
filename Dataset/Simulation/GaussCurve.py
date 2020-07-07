import numpy as np

def GaussCurve(x, I, C, W):
    return I*np.exp(-np.sqrt(2)/8*((x-C)/W)**2)


def FBG_spectra(x_coord, X):
    x_coord = np.tile(x_coord, X.shape+(1,))
    X = np.expand_dims(X, axis=len(X.shape))
    I = np.array([5.72, 2.95, 2.2, 1, 0.5])[:, np.newaxis]*0.001
    return np.sum(GaussCurve(x_coord, I, X, 0.05), axis=1)