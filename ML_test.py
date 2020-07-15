from Model import SignalError
from Dataset.Simulation.GaussCurve_TF import FBG_spectra
import tensorflow as tf
import matplotlib.pyplot as plt
e_model, f_model = SignalError.ErrorModel()

samples = 1000
fbgs = 5

def normalize(spectra):
    maximum = tf.reduce_max(spectra, axis=1)
    minimum = tf.reduce_max(spectra, axis=1)
    return (spectra-minimum)/(maximum-minimum)

x_coord = tf.linspace(0.0, 1.0, 1000)

X1 = tf.random.uniform([samples, fbgs])
I1 = tf.random.uniform([samples, fbgs], 0.1, 1)
W1 = tf.random.uniform([samples, fbgs], 0.1, 0.3)
spectrums1 = normalize(FBG_spectra(x_coord, X1, I1, W1))

X2 = tf.random.uniform([samples, fbgs])
I2 = tf.random.uniform([samples, fbgs], 0.1, 1)
W2 = tf.random.uniform([samples, fbgs], 0.1, 0.3)
spectrums2 = normalize(FBG_spectra(x_coord, X2, I2, W2))

train_X = tf.concat([tf.expand_dims(spectrums1, axis=1),
                     tf.expand_dims(spectrums2, axis=1)], axis=1)
                
train_Y = tf.reduce_mean(tf.abs(X2-X1), axis=1)

# plt.plot(spectrums1[0])
# plt.plot(spectrums2[0])
# plt.show()

e_model.summary()
e_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse")
e_model.fit(train_X, train_Y, epochs = 5, batch_size=256)
e_model.save_weights('./SavedModel/SignalErrorModel.hdf5')
pred_Y = e_model(train_X)
plt.plot(pred_Y-train_Y)
plt.show()
