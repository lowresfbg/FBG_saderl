from Solution.DE_general import DE
from Dataset.loader import DATASET_3fbg_1
import tensorflow as tf

from Tests.GeneralTest_logger import CompleteTester, DatasetConfig, TestConfig
import json

import matplotlib.pyplot as plt
import numpy as np

from AutoFitAnswer import Get3FBGAnswer

from TestResult.loader import ResultSet_Iter200, ResultSet_EarlyStop, ResultSet_CR, ResultSet_FWHM


def F_test():
    dataset = tf.constant(DATASET_3fbg_1(), dtype=tf.dtypes.float32)
    # for CR in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    # for F in [0.1,0.5,0.8,1,1.2,1.5,2,2.5,3]:
    for FWHM_multiplier in [0.5,0.75,0.8,0.9,1,1.1,1.5]:
        de = DE()
        # de.EarlyStop = False
        name = "FWHM_multiplier{:03d}".format(int(FWHM_multiplier*100))
        # name = "F{:02d}".format(int(F*10))

        dataset_config = DatasetConfig(dataset)
        dataset_config.W *= FWHM_multiplier

        tester = CompleteTester(name,
                                dataset_config,
                                TestConfig(50, 1, 0.7), de)
        # tester.max_iter = 200
        tester.run()
        data = tester.GetData()

        # print(data)
        # with open("./TestResult/FWHM_mul/{}.json".format(name), 'w') as f:
        #     json.dump(data, f)

F_test()


def plot_result():
    rs = ResultSet_FWHM()

    answer = np.array(Get3FBGAnswer()).T
    plt.subplot(211)
    plt.plot(answer)

    print(answer.shape)

    X_Data = []
    Y_Data = []
    for i, r in enumerate(rs):

        FWHM_M = [0.5,0.75,0.8,0.9,1,1.1,1.5][i]

        iterations = []

        for td in r["test_data"]:
            iterations.append(td["iteration"])

        print(np.array(r["X_log"]).shape, answer.shape)

        err_to_answers = np.sqrt(
            np.mean((np.array(r["X_log"]) - answer)**2, axis=1))

        # label = "CR={}".format(r["config"]["CR"])
        label = "FWHM_M={}".format(FWHM_M)

        plt.subplot(221)
        plt.grid()
        plt.plot(np.array(r["X_log"]))
        plt.subplot(222)
        plt.yscale("log")
        plt.grid()
        plt.plot(iterations, '-o', label=label)

        plt.subplot(223)
        plt.yscale("log")
        plt.plot(err_to_answers, '-o', label=label)
        plt.grid()

        X_Data.append(FWHM_M)
        Y_Data.append(np.mean(err_to_answers))

    plt.legend()
    plt.subplot(224)
    plt.yscale("log")
    plt.plot(X_Data, Y_Data, "-o")
    plt.grid()

    plt.show()


plot_result()
