import numpy as np


def Mutate(X, CR=0.5, F=1):
    NP = len(X)

    i = np.repeat([np.arange(NP)], NP, axis=0)
    i1 = np.repeat([np.arange(NP-1)], NP, axis=0)
    i2 = np.repeat([np.arange(NP-2)], NP, axis=0)
    i3 = np.repeat([np.arange(NP-3)], NP, axis=0)
    a = np.repeat(np.random.randint(NP-1, size=NP)
                  [:, np.newaxis], NP-1, axis=1)
    b = np.repeat(np.random.randint(NP-2, size=NP)
                  [:, np.newaxis], NP-2, axis=1)
    c = np.repeat(np.random.randint(NP-3, size=NP)
                  [:, np.newaxis], NP-3, axis=1)

    remain = i[i != i.T].reshape(NP, -1)
    ai = remain[i1 == a]
    remain = remain[i1 != a].reshape(NP, -1)
    bi = remain[i2 == b]
    remain = remain[i2 != b].reshape(NP, -1)
    ci = remain[i3 == c]

    V = X[ai] + F*(X[bi]-X[ci])

    X = X + ((V-X)*(np.random.rand(*X.shape) < CR))

    return X


