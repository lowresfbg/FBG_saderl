
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt

from Solution.DE_general import DE, spectra_diff_contrast, spectra_diff_absolute

from Dataset.Simulation.GaussCurve_TF import FBG_spectra, GaussCurve

# DATASET!!!!!!!!!!!!!
from Dataset.loader import DATASET_3fbg_1_2 as Dataset
from Dataset.loader import DATASET_background
from Dataset import Resampler

from Algorithms.PeakFinder import FindPeaks

from Tests.GeneralTest_logger import SingleLogger
import tensorflow as tf

import numpy as np
from AutoFitAnswer import Get3FBGAnswer


import matplotlib

cmap = matplotlib.cm.get_cmap('cool')


print('loading dataset')
# [:,:,948-76:1269-76]

fbgs = 10
samples = 1000



x_coord = tf.linspace(1545.0, 1549.0, 1000)

X1 = tf.random.uniform([samples, fbgs])*3+1545.5
I1 = tf.repeat([[ 1, 0.5, 0.3, 0.2, 0.1 ,0.05,0.03,0.02,0.01,0.005 ]], samples, axis=0)*1e3
W1 = tf.ones([samples, fbgs]) * 0.2
spectrums1 = FBG_spectra(x_coord, X1, I1, W1) 
spectrums1 = spectrums1 + (tf.random.uniform(spectrums1.shape)-0.5) * 0.001

dataset =tf.concat([ tf.repeat(x_coord[tf.newaxis,tf.newaxis, :], samples, axis=0), tf.expand_dims(spectrums1, axis=1) ] , axis=1)

print(dataset)
print('loading completed')

plt.plot(*dataset[0])
plt.show()
ITERATION = 2000

de = DE()

def InsertAnswer(info, answer):
    i, data, X = info
    amount = 25
    if i<1:
        de.X = tf.concat([ tf.repeat([answer],amount, axis=0) + (tf.random.uniform([amount, fbgs])-0.5)*2*0.001 , de.X[amount:]], axis=0)



def giveAnswerGenerator(answer):
    return lambda info : InsertAnswer(info, answer)


I = I1[0]
W = W1[0]

def init(de, answer, giveAnswer = False):
    de.minx = 1545.0
    de.maxx = 1549.0
    de.I = I
    de.W = W
    de.NP = 50
    de.CR = 0.75
    de.F = 0.5
    # de.Ranged = True
    de.EarlyStop_threshold = 1e-3
    # de.spectra_diff = spectra_diff_contrast
    de.beforeEach = []

    if giveAnswer:
        de.beforeEach.append(giveAnswerGenerator(answer))




sl = SingleLogger(I, W)

print("start single test")
init(de, X1[3])
de.afterEach.append(sl.log)
de.run(dataset[3], max_iter=ITERATION)
sl.PlotFigure()
plt.show()
print("single test complete")


X_log = []
err_log = []

de.afterEach.clear()
print(x_coord)
print(spectrums1[0])

data = tf.concat([
    [x_coord],
    [spectrums1[0]]
], axis=0)
print(data.shape)


input_error_rmse_log = []
iterations_log = []
output_error_rmse_log = []
c = []

errorOverIterationsLine = []


for rt in range(10):
    for i in range(spectrums1.shape[0]):
        for j in range(2):
            sl = SingleLogger(I, W)
            sl.Animate = False


            input_error = (tf.random.uniform([fbgs])-0.5)*2*0.001
            giveAnswer = j%2==0

            input_error_rmse = tf.sqrt(tf.reduce_mean(input_error**2))
            input_error_rmse_log.append(input_error_rmse)

            data = tf.concat([
                [x_coord],
                [spectrums1[i]]
            ], axis=0)

            print('run ------------------')

            answer = X1[i]
            init(de, answer+input_error, giveAnswer)
            de.afterEach = [sl.log]

            X = de(data, max_iter=ITERATION)

            x_mean = tf.reduce_mean(X, axis=0)

            iterations_log.append(de.iter)
            output_error_rmse = tf.sqrt(tf.reduce_mean((x_mean-answer)**2))
            output_error_rmse_log.append(output_error_rmse)

            errorOverIterationsLine.append(np.sqrt(np.mean((np.array(sl.X_mean_log)-np.array(answer)[np.newaxis, :])**2,axis=1)))
            

            if giveAnswer:
                c.append(cmap(input_error_rmse*1000))
            else:
                c.append((1,.6,.1))

            plt.clf()

            plt.xscale('log')
            plt.yscale('log')

            plt.scatter(np.array(iterations_log),np.array(output_error_rmse_log), c=c, s=4)

            for j,line in enumerate(errorOverIterationsLine):
                plt.plot(line, c=c[j], alpha=0.1)


            plt.pause(0.01)
            plt.tight_layout()


plt.show()