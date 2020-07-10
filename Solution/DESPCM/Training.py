from Model.SpectrumCompensate import SpectrumCompensateModel
from Solution.DESPCM import DESPCM
import tensorflow as tf
# Spectrum, SCM = SpectrumCompensateModel.GetModel()


def SCM_input_training(X, I, W):
    # intensity compensation
    I = tf.repeat([I], X.shape[0], axis=0)
    I = tf.concat([I[:, :1] + (X[:, :1] - 1546.52)
                   * -0.35, I[:, 1:]], axis=1)
    W = tf.repeat([W], X.shape[0], axis=0)

    I = tf.expand_dims(I, axis=2)
    X = tf.expand_dims(X, axis=2)
    W = tf.expand_dims(W, axis=2)
    # SCM input
    input_fbg = tf.concat([I, X, W], axis=2)

    return input_fbg


def newloss(y1, y2):
    return tf.reduce_mean((y1-y2)**2/(y1+y2))


def Compile(SCM):
    optimizer = tf.optimizers.Adam(lr=2e2)
    SCM.compile(optimizer=optimizer, loss="mse")


def Train(SCM, dataset, X, I, W):
    input_fbg = SCM_input_training(X, I, W)
    input_x = dataset[:, 0]
    SCM.fit((input_x, input_fbg), dataset[:, 1], epochs=200, batch_size=1000)
