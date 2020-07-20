# DE
from Solution.DE_general import DE, spectra_diff_contrast, spectra_diff_absolute, Evaluate
import csv

# ML
from Model import WavelengthCNN
from Dataset.Simulation.GaussCurve_TF import FBG_spectra
import tensorflow as tf
import matplotlib.pyplot as plt
from AutoFitAnswer import GetFBGAnswer

import numpy as np

from Dataset.loader import DATASET_5fbg_1_perfect as Dataset
from Algorithms.PeakFinder import FindPeaks

from AutoFitAnswer import Get3FBGAnswer

# resample
from scipy.interpolate import interp1d

threshold = 1.5e-5  # 3fbg-2
fbg_count = 5
# threshold =  1.5e-5 # 3fbg-2


def Resample(skip=1, target=1000):
    dataset = Dataset()

    if type(dataset) is tuple:
        print('is tuple!')
        dataset, answertable = dataset
        answer = answertable[:, :fbg_count]
        peaks = tf.constant(np.concatenate([
            answertable[0, fbg_count:fbg_count*2, np.newaxis],
            answertable[0, :fbg_count, np.newaxis],
            answertable[0, fbg_count*2:, np.newaxis],
        ], axis=1), dtype=tf.dtypes.float32)
    else:
        answer = np.array(GetFBGAnswer(dataset, 3, threshold)).T
        peaks = tf.constant(
            FindPeaks(dataset[0], threshold), dtype=tf.dtypes.float32)

    dataset = tf.constant(dataset, dtype=tf.dtypes.float32)[:, :, ::skip]
    x_coord = np.linspace(np.min(dataset[:, 0]), np.max(dataset[:, 0]), target)
    new_dataset = []
    for i in range(dataset.shape[0]):
        new_dataset.append(
            interp1d(dataset[i, 0], dataset[i, 1],  kind='cubic')(x_coord))
    dataset = tf.cast(tf.concat([
        tf.repeat([[x_coord]], dataset.shape[0], axis=0),
        np.expand_dims(new_dataset, axis=1)], axis=1), tf.dtypes.float32)

    return dataset, answer, peaks


dataset, answer, peaks = Resample(1, 10000)
# plt.plot(dataset[0][1])
# plt.show()


# dataset = tf.constant(dataset, dtype=tf.dtypes.float32)

# print(dataset)


# peaks = tf.constant(FindPeaks(dataset[0], 1e-5), dtype=tf.dtypes.float32)
# peaks = tf.constant(FindPeaks(dataset[0], 1.3e-4), dtype=tf.dtypes.float32)

fbgs = len(peaks)

I = peaks[:, 2]*1e3
W = peaks[:, 0]

model = WavelengthCNN.GetModel(fbgs)

ITERATION = 1000

de = DE()


def init(de):
    de.minx = 1545.0
    de.maxx = 1549.0
    de.I = I
    de.W = W
    de.NP = 50
    de.CR = 0.7
    de.F = 0.99
    de.EarlyStop_threshold = 4e-3
    de.spectra_diff = spectra_diff_absolute


init(de)


X_log = []
iter_log = []


def evaluateData(data):
    print('run ------------------')
    X = de(data, max_iter=ITERATION)
    Xmean = tf.reduce_mean(X, axis=0)
    X_log.append(Xmean)

    # err = Evaluate(data, tf.expand_dims(Xmean, axis=0), I, W, spectra_diff_absolute)[0]
    iter_log.append(de.iter)

    plt.clf()

    plt.plot(answer, "--", color='gray')
    for i in range(X_log[0].shape[0]):
        plt.plot(np.array(X_log)[:, i], "o-", label="FBG{}".format(i+1))

    # plt.twinx()
    # plt.plot(err_log)
    plt.legend()
    plt.xlabel("Test set")
    plt.ylabel("Wavelength")
    plt.pause(0.01)
    plt.tight_layout()
    return Xmean

# MACHINE LEARNING!!!


def normalize(spectra):
    maximum = tf.expand_dims(tf.reduce_max(spectra, axis=1), axis=1)
    minimum = tf.expand_dims(tf.reduce_min(spectra, axis=1), axis=1)
    return (spectra-minimum)/(maximum-minimum)


dataset, answer, peaks = Resample(1, 1000)

train_X = normalize(dataset[:, 1])
test_X = normalize(dataset[:, 1])
RUN = True

if RUN:
    train_Y = tf.map_fn(evaluateData, dataset)
else:
    with open('./temp_result.csv', 'r') as f:
        lst = [line for line in list(csv.reader(f)) if len(line) > 0]
        array = np.array(lst).astype(float)
        train_Y = tf.constant(array, dtype=tf.dtypes.float32)

with open('./temp_result.csv', 'w') as f:
    csv.writer(f).writerows(train_Y.numpy().tolist())

plt.show()

# plt.plot(train_Y)
# plt.show()


model.compile(optimizer=tf.keras.optimizers.Adam(lr=5e-4), loss="mse")


def plot_ml(batch, logs):
    if batch % 2 == 0:
        pred_Y = model(train_X)+1545
        plt.clf()
        plt.plot(answer, "--", color='gray')

        for i in range(pred_Y[0].shape[0]):
            plt.plot(np.array(pred_Y)[:, i], "-", label="ML-FBG{}".format(i+1))

        for i in range(train_Y[0].shape[0]):
            plt.plot(np.array(train_Y)[:, i], "o",
                     label="DE-FBG{}".format(i+1))
        # plt.plot(pred_Y, "-", label="ML")
        # plt.plot(train_Y, "o", label="DE")

        plt.title(batch)
        plt.xlabel("Test set")
        plt.ylabel("Wavelength")
        plt.legend()

        plt.pause(0.02)


sample_weight = tf.constant(iter_log, dtype=tf.dtypes.float32)
sample_weight_max = tf.reduce_max(sample_weight)
sample_weight_min = tf.reduce_min(sample_weight)
sample_weight = (sample_weight - sample_weight_max) / \
    (sample_weight_min-sample_weight_max)
sample_weight = sample_weight*0.8+0.2

model.fit(train_X, train_Y-1545, epochs=300, batch_size=2000, validation_split=0.0, shuffle=True,
          sample_weight=sample_weight,
          callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=plot_ml)])
pred_Y = model(test_X)+1545
plt.plot(pred_Y, "-")
plt.plot(train_Y, "o")


with open('./temp_result_ML.csv', 'w') as f:
    csv.writer(f).writerows(pred_Y.numpy().tolist())

error = np.sqrt(np.mean((train_Y - answer)**2))
print("DE_error", error)

error = np.sqrt(np.mean((pred_Y - answer)**2))
print("ML_error", error)

plt.show()
