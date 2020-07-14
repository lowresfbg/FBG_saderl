import tensorflow as tf
from Dataset.Simulation.GaussCurve_TF import GaussCurve_Graph


def CompensateModel():
    input_layer = tf.keras.layers.Input((4,))
    x = input_layer
    x = tf.keras.layers.Dense(5, activation="sigmoid")(x)
    x = tf.keras.layers.Dense(1)(x)
    # print(x, input_layer)
    # x+=tf.math.sin(input_layer[:,:1]*200)
    x *= tf.exp(-(input_layer[:, :1]/0.23/20)**2*tf.math.log(2.0)*4)
    return tf.keras.Model(input_layer, x)


def GetModel():
    input_xcoord = tf.keras.layers.Input((None,))
    input_fbg = tf.keras.layers.Input((None, 3))

    X = tf.expand_dims(input_xcoord, axis=1)

    I = tf.expand_dims(input_fbg[:, :, 0], axis=2)
    C = tf.expand_dims(input_fbg[:, :, 1], axis=2)
    W = tf.expand_dims(input_fbg[:, :, 2], axis=2)

    gauss = GaussCurve_Graph(X, I, C, W)

    spectra = tf.reduce_sum(gauss, axis=1)*0.001

    # compensation
    CM = CompensateModel()

    full_shape = tf.shape(gauss)

    X_flat = tf.reshape(tf.broadcast_to(X, full_shape), (-1, 1))
    I_flat = tf.reshape(tf.broadcast_to(I, full_shape), (-1, 1))
    C_flat = tf.reshape(tf.broadcast_to(C, full_shape), (-1, 1))
    W_flat = tf.reshape(tf.broadcast_to(W, full_shape), (-1, 1))

    CM_input = tf.concat([(X_flat-C_flat)*20.0, I_flat*1000*0,
                          (C_flat-1545.0)/10*0 , W_flat/0.2*0], axis=1)

    compensates = CM(CM_input) 

    compensates = tf.reshape(compensates, full_shape)*I*0.001

    compensates = tf.reduce_sum(compensates, axis=1)

    compensated = compensates + spectra

    spectra_model = tf.keras.Model(
        inputs=(input_xcoord, input_fbg), outputs=spectra)
    scm_model = tf.keras.Model(
        inputs=(input_xcoord, input_fbg), outputs=compensated)

    return spectra_model, scm_model
