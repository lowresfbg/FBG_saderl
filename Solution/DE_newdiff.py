from Solution.DE import DE
import numpy as np

class DE_newdiff(DE):
    def spectra_diff(self, A, B):
        return np.mean((A-B)**2/(A+B), axis=1)