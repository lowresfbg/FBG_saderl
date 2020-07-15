import numpy as np
import tensorflow as tf

@tf.function
def GaussCurve(x, I, C, W):
    return I*tf.exp(-tf.math.log(2.0)*4*((x-C)/W)**2)

@tf.function
def FBG_spectra(x_coord, X, I, W):
    x_coord = tf.cast(tf.tile([[x_coord]], X.shape+(1,)),tf.dtypes.float32)
    X = tf.expand_dims(X, axis=len(X.shape))
    I = tf.expand_dims(I, axis=len(I.shape))
    W = tf.expand_dims(W, axis=len(W.shape))
    I*=0.001

    return tf.reduce_sum(GaussCurve(x_coord, I, X, W), axis=1)

def GaussCurve_Graph(x, I, C, W):
    return I*tf.exp(-tf.sqrt(2.0)*4*((x-C)/W)**2)

# def FBG_spectra_each(x_coord, X, I, W):
#     x_coord = tf.cast(tf.tile([[x_coord]], X.shape+(1,)),tf.dtypes.float32)
#     return tf.reduce_sum(GaussCurve(x_coord, I, X, W), axis=1)
