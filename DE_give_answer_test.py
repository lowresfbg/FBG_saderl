from cycler import cycler
default_cycler = (cycler(color=[
    '#3f51b5',
    '#ff5722',
    '#4caf50',
    '#e91e63',
    '#9c27b0',
    '#2196f3',
    '#fbc02d']))
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = default_cycler




import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

import json

print('loading dataset')
# [:,:,948-76:1269-76]

samples = 20
testRepeat = 5
ITERATION = 3000

errorOverIterationsLineGroup = []
fbgsCounts = [3,5,7]

try:
    with open('./DE_give_answer_test_result', 'r') as f:
        errorOverIterationsLineGroup = [[np.array(line) for line in group] for group in json.load(f)]
except:
    pass

plt.figure(figsize=(3.89*2,3.98), dpi=150)
def plotAll():
    plt.clf()
    plt.xlabel("Iterations\n"+ r"$\bf{(a)}$")
    plt.ylabel("RMSE (nm)")
    plot()
    plt.tight_layout()

def plot():
    with open('./DE_give_answer_test_result', 'w', newline='') as f:
        print(errorOverIterationsLineGroup)
        json.dump([[line.tolist() for line in group] for group in errorOverIterationsLineGroup], f)

    average = np.zeros([len(fbgsCounts),2,ITERATION])
    n_of_line = np.zeros([len(fbgsCounts),2,ITERATION])
    std = np.zeros([len(fbgsCounts),2,ITERATION])

    def padTo(array, length):
        return np.concatenate([array, np.zeros(length-array.shape[0])], axis=0)

    for (i,errorOverIterationsLine) in enumerate(errorOverIterationsLineGroup):
        for (j,line) in enumerate(errorOverIterationsLine):
            ga = j%2
            average[i,ga]+=padTo(line, ITERATION)
            n_of_line[i,ga] += padTo(line*0+1, ITERATION)

    average /= n_of_line

    for (i,errorOverIterationsLine) in enumerate(errorOverIterationsLineGroup):
        for (j,line) in enumerate(errorOverIterationsLine):
            ga = j%2
            std[i,ga] += (padTo(line, ITERATION) - average[i,ga])**2

    std /= n_of_line
    std = np.sqrt(std)
    print(std)

    plt.xscale('log')
    plt.yscale('log')

    for j in range(2):
        for (i,fbgs) in enumerate(fbgsCounts):
            a = average[i,j]
            d = std[i,j]
            ga = j%2==1
            label = "{}FBG".format(fbgs)
            if ga:
                label+="-AECNNDE"
            else:
                label+="-DE"
            p = plt.plot(a, label=label)
            plt.fill_between(np.arange(ITERATION), a-d, a+d, facecolor=p[0].get_color(), alpha=0.5)


    plt.legend(fontsize=6, ncol=2, loc='lower left')
            
plotAll()

def calculate():
    global errorOverIterationsLineGroup
    errorOverIterationsLineGroup = []
    for fbgs in fbgsCounts:

        x_coord = tf.linspace(1545.5, 1548.5, 1000)

        X1 = tf.random.uniform([samples, fbgs])*3+1545.5
        I1 = tf.repeat([[ 1, 0.5, 0.25, 0.125, 0.0625 ,0.03125,0.015625,0.0078125,0.000390625,0.000001953125 ][:fbgs]], samples, axis=0)*1e3
        W1 = tf.ones([samples, fbgs]) * 0.15
        
        spectrums1 = FBG_spectra(x_coord, X1, I1, W1) 
        # spectrums1 = spectrums1 + (tf.random.uniform(spectrums1.shape)-0.5) * 0.0001

        dataset =tf.concat([ tf.repeat(x_coord[tf.newaxis,tf.newaxis, :], samples, axis=0), tf.expand_dims(spectrums1, axis=1) ] , axis=1)


        print(dataset)
        print('loading completed')

        # plt.plot(*dataset[0])
        # plt.show()
        

        de = DE()

        def InsertAnswer(info, answer):
            i, data, X = info
            amount = fbgs*10
            if i<1:
                de.X = tf.concat([ tf.repeat([answer],amount, axis=0) + (tf.random.uniform([amount, fbgs])-0.5)*2*0.002 , de.X[amount:]], axis=0)



        def giveAnswerGenerator(answer):
            return lambda info : InsertAnswer(info, answer)


        I = I1[0]
        W = W1[0]

        def init(de, answer, giveAnswer = False):
            de.minx = 1545.0
            de.maxx = 1549.0
            de.I = I
            de.W = W
            de.NP = fbgs*20
            de.CR = 0.9
            de.F = 0.8
            de.Ranged = True
            de.EarlyStop_threshold = 5e-3
            # de.spectra_diff = spectra_diff_contrast
            de.beforeEach = []

            if giveAnswer:
                de.beforeEach.append(giveAnswerGenerator(answer))




        # sl = SingleLogger(I, W)

        # print("start single test")
        # init(de, X1[3])
        # de.afterEach.append(sl.log)
        # de.run(dataset[3], max_iter=ITERATION)
        # sl.PlotFigure()
        # plt.show()
        # print("single test complete")


        # X_log = []
        # err_log = []

        # de.afterEach.clear()
        # print(x_coord)
        # print(spectrums1[0])

        # data = tf.concat([
        #     [x_coord],
        #     [spectrums1[0]]
        # ], axis=0)
        # print(data.shape)


        # input_error_rmse_log = []
        # iterations_log = []
        # output_error_rmse_log = []
        # c = []

        errorOverIterationsLine = []


        for rt in range(testRepeat):
            for i in range(spectrums1.shape[0]):
                for j in range(2):
                    sl = SingleLogger(I, W)
                    sl.Animate = False


                    input_error = (tf.random.uniform([fbgs])-0.5)*2*0.001
                    giveAnswer = j%2==1

                    # input_error_rmse = tf.sqrt(tf.reduce_mean(input_error**2))
                    # input_error_rmse_log.append(input_error_rmse)

                    data = tf.concat([
                        [x_coord],
                        [spectrums1[i]]
                    ], axis=0)

                    print('run ------------------')

                    answer = X1[i]
                    init(de, answer+input_error, giveAnswer)
                    de.afterEach = [sl.log]

                    X = de(data, max_iter=ITERATION)

                    # x_mean = tf.reduce_mean(X, axis=0)

                    # iterations_log.append(de.iter)
                    # output_error_rmse = tf.sqrt(tf.reduce_mean((x_mean-answer)**2))
                    # output_error_rmse_log.append(output_error_rmse)

                    errorOverIterationsLine.append(np.sqrt(np.mean((np.array(sl.X_mean_log)-np.array(answer)[np.newaxis, :])**2,axis=1)))
                    

                    # if giveAnswer:
                    #     c.append(cmap(input_error_rmse*1000))
                    # else:
                    #     c.append((1,.6,.1))

                    # plt.clf()

                    # plt.xscale('log')
                    # plt.yscale('log')

                    # plt.scatter(np.array(iterations_log),np.array(output_error_rmse_log), c=c, s=4)

                    # for j,line in enumerate(errorOverIterationsLine):
                    #     plt.plot(line, alpha=0.1)


                    # plt.pause(0.01)
                    # plt.tight_layout()

        errorOverIterationsLineGroup.append(errorOverIterationsLine)
        print(errorOverIterationsLineGroup)

        plotAll()
        plt.pause(0.01)
    print('done')

# calculate()
plt.show()


