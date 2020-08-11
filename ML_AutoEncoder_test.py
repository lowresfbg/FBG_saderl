from Model import AutoEncoderWLCNN
from Dataset.Simulation.GaussCurve_TF import FBG_spectra
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from Dataset.loader import DATASET_5fbg_1_perfect as Dataset
from Dataset import Resampler

fbgs = 5
samples = 10000


def normalize(spectra):
    maximum = tf.expand_dims(tf.reduce_max(spectra, axis=1), axis=1)
    minimum = tf.expand_dims(tf.reduce_min(spectra, axis=1), axis=1)
    return (spectra-minimum)/(maximum-minimum)

x_coord = tf.linspace(0.0, 1.0, 1000)

X1 = tf.random.uniform([samples, fbgs])
I1 = tf.repeat([[1,0.6,0.3,0.2,0.1]], samples, axis=0)
W1 = tf.ones([samples, fbgs]) * 0.05
spectrums1 = normalize(FBG_spectra(x_coord, X1, I1, W1))


train_X = spectrums1
train_Y = spectrums1
train_Y_wl = X1 + 1545

# load from data set
LOAD_DATASET = False
if LOAD_DATASET:
    dataset, answer, peaks = Resampler.Resample(Dataset(), fbgs)
    spectrums1 = normalize(dataset[:, 1])
    x_coord = dataset[0, 0]

    maxy = np.max(spectrums1[0])

    train_X = spectrums1 / maxy

    train_Y = spectrums1
    train_Y_wl = answer

# ------------

# plt.plot(spectrums1[0])
# plt.show()


def test(AE=True):
    encdec, model, enc, dec = AutoEncoderWLCNN.GetModel(fbgs)

    # encdec.summary()
    if AE:
        encdec.load_weights('./SavedModel/EncDecModel.hdf5')

        # for layer in encdec.layers:
        #     layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse")


    class ML_logger:
        def __init__(self, model):
            self.model = model
            self.loss_log = []
            self.val_loss_log = []

        def plot_ml(self,batch, logs):

            # pred_Y = self.model(train_X)+1545
            # self.err_log.append(np.mean((pred_Y-train_Y_wl)**2))

            self.loss_log.append(logs['loss'])
            self.val_loss_log.append(logs['val_loss'])

            # if batch % 10 == 0:

            #     plt.clf()
            #     plt.subplot(211)

            #     plt.plot(answer, "--", color='gray')

            #     for i in range(pred_Y[0].shape[0]):
            #         plt.plot(np.array(pred_Y)[:, i], "-", label="ML-FBG{}".format(i+1))

            #     for i in range(train_Y_wl[0].shape[0]):
            #         plt.plot(np.array(train_Y_wl)[:, i], "o",
            #                 label="DE-FBG{}".format(i+1))
            #     # plt.plot(pred_Y, "-", label="ML")
            #     # plt.plot(train_Y_wl, "o", label="DE")

            #     plt.title(batch)
            #     plt.xlabel("Test set")
            #     plt.ylabel("Wavelength")
            #     plt.legend()

            #     plt.subplot(212)

            #     plt.plot(self.err_log)
            #     plt.yscale('log')

            #     plt.pause(0.02)

    logger = ML_logger(model)

    # print("training cycle", i)

    model.fit(train_X, train_Y_wl-1545, epochs=2000, batch_size=10000,
            validation_split = 0.9, verbose=1,
            callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=logger.plot_ml)])
    # model.save_weights('./SavedModel/EncDecWLModel.hdf5')

    # pred_Y = encdec(train_X)
    # pred_Y_wl = model(train_X)+1545
    # print(pred_Y.shape, train_Y.shape)

    # for i in range(samples):
    #     plt.plot(x_coord, pred_Y[i], "o")
    #     plt.plot(x_coord, train_Y[i], "-")
    #     for j in list(pred_Y_wl[i]):
    #         plt.axvline(j.numpy())
    #     plt.show()
    return logger.loss_log, logger.val_loss_log

plt.figure(figsize=(3.89*2,3.98), dpi=150)


plt.subplot(211)
plt.yscale('log')
plt.xlabel("Epochs\n"+ r"$\bf{(b)}$")
plt.ylabel("RMSE (nm)")
plt.grid(linestyle=":")
plt.subplot(212)
plt.yscale('log')
plt.xlabel("Epochs\n"+ r"$\bf{(c)}$")
plt.ylabel("RMSE (nm)")
plt.grid(linestyle=":")

calculate = False

import csv




for i in range(10):
    with open('./AETestResults/fitting{:02d}.csv'.format(i), 'r') as f:
        data = [np.array(line).astype(float) for line in list(csv.reader(f))]

    print(data)

    with open('./AETestResults/fitting{:02d}.csv'.format(i), 'w', newline='') as f:

        writer = csv.writer(f)

        # red for AE
        if calculate:
            t = test(True)
        else:
            t = data[:2]

        plt.subplot(211)
        plt.plot(t[0], c='#ff5722', alpha=0.4, label="with AE") # 0 for train
        plt.subplot(212)
        plt.plot(t[1], c='#ff5722', alpha=0.4, label="with AE") # 1 for test
        plt.tight_layout()
        plt.pause(0.01)

        writer.writerow(t[0])
        writer.writerow(t[1])

        # blue for compare
        if calculate:
            t = test(False)
        else:
            t = data[2:]
            
        plt.subplot(211)
        plt.plot(t[0], c='#2196f3', alpha=0.4, label="without AE") # 0 for train
        plt.subplot(212)
        plt.plot(t[1], c='#2196f3', alpha=0.4, label="without AE") # 1 for test
        plt.tight_layout()
        plt.pause(0.01)

        writer.writerow(t[0])
        writer.writerow(t[1])


plt.show()
