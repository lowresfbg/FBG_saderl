from Model import AutoEncoderWLCNN
from Dataset.Simulation.GaussCurve_TF import FBG_spectra
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from Dataset.loader import DATASET_3fbg_1_2
from Dataset import Resampler

fbgs = 5
samples = 20

encdec, model = AutoEncoderWLCNN.GetModel(3)


def normalize(spectra):
    maximum = tf.expand_dims(tf.reduce_max(spectra, axis=1), axis=1)
    minimum = tf.expand_dims(tf.reduce_min(spectra, axis=1), axis=1)
    return (spectra-minimum)/(maximum-minimum)


x_coord = tf.linspace(0.0, 1.0, 1000)

X1 = tf.random.uniform([samples, fbgs])
I1 = tf.random.uniform([samples, fbgs])
W1 = tf.ones([samples, fbgs]) * tf.random.uniform([1], 0.05, 0.15)
spectrums1 = normalize(FBG_spectra(x_coord, X1, I1, W1))

dataset, answer, peaks = Resampler.Resample(DATASET_3fbg_1_2(), 3)
spectrums1 = normalize(dataset[:, 1])
x_coord = dataset[0, 0]

train_X = spectrums1

train_Y = spectrums1
train_Y_wl = answer
# plt.plot(spectrums1[0])
# plt.show()

encdec.summary()
# encdec.load_weights('./SavedModel/EncDecModel.hdf5')

# for layer in encdec.layers:
#     layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-2), loss="mse")


class ML_logger:
    def __init__(self, model):
        self.model = model
        self.err_log = []

    def plot_ml(self,batch, logs):
        if batch % 5 == 0:
            pred_Y = self.model(train_X)+1545
            plt.clf()
            plt.subplot(211)

            plt.plot(answer, "--", color='gray')

            for i in range(pred_Y[0].shape[0]):
                plt.plot(np.array(pred_Y)[:, i], "-", label="ML-FBG{}".format(i+1))

            for i in range(train_Y_wl[0].shape[0]):
                plt.plot(np.array(train_Y_wl)[:, i], "o",
                        label="DE-FBG{}".format(i+1))
            # plt.plot(pred_Y, "-", label="ML")
            # plt.plot(train_Y_wl, "o", label="DE")

            plt.title(batch)
            plt.xlabel("Test set")
            plt.ylabel("Wavelength")
            plt.legend()

            plt.subplot(212)
            self.err_log.append(np.mean((pred_Y-train_Y_wl)**2))
            plt.plot(self.err_log)
            plt.yscale('log')

            plt.pause(0.02)
logger = ML_logger(model)

# print("training cycle", i)

model.fit(train_X[::2], train_Y_wl[::2]-1545, epochs=500, batch_size=200, shuffle=True,
          callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=logger.plot_ml)])
model.save_weights('./SavedModel/EncDecWLModel.hdf5')

# pred_Y = encdec(train_X)
# pred_Y_wl = model(train_X)+1545
# print(pred_Y.shape, train_Y.shape)


# for i in range(samples):
#     plt.plot(x_coord, pred_Y[i], "o")
#     plt.plot(x_coord, train_Y[i], "-")
#     for j in list(pred_Y_wl[i]):
#         plt.axvline(j.numpy())
#     plt.show()

plt.show()
