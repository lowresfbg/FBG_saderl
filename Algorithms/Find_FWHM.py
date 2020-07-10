import numpy as np
import csv
import matplotlib.pyplot as plt

def tofind(data, up, down):                        #吃進一個波的範圍
    X = data[0][up:down]
    Y = data[1][up:down]
    return X, Y

def findFWHM(data, up,down):
    X, Y = tofind(data, up, down)

    flag = 0
    FWHP1 = 0
    FWHP2 = 0
    Inte = 0
    posi = 0
    posi1 = 0
    posi2 = 0
    for i in range(0,len(X)):
        if ((max(Y)/2 - Y[i]) < 1e-10):         #找半波位置
            if (flag == 0):
                FWHP1 = X[i]
                posi1 = i
                flag = flag + 1
            else:
                FWHP2 = X[i]
                posi2 = i

    FWHM = FWHP2-FWHP1
    posi = X[int((posi1+posi2)/2)]              #找半波高&位置
    Inte = Y[int((posi1+posi2)/2)]
    