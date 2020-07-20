from Model import AutoEncoderWLCNN
from Dataset.Simulation.GaussCurve_TF import FBG_spectra
import tensorflow as tf
import matplotlib.pyplot as plt

fbgs = 5
samples = 20

encdec, model = AutoEncoderWLCNN.GetModel(5)



def normalize(spectra):
    maximum = tf.expand_dims(tf.reduce_max(spectra, axis=1), axis=1)
    minimum = tf.expand_dims(tf.reduce_min(spectra, axis=1), axis=1)
    return (spectra-minimum)/(maximum-minimum)


x_coord = tf.linspace(0.0, 1.0, 1000)

X1 = tf.random.uniform([samples, fbgs])
I1 = tf.random.uniform([samples, fbgs])
W1 = tf.ones([samples, fbgs]) * tf.random.uniform([1], 0.05, 0.15)
spectrums1 = normalize(FBG_spectra(x_coord, X1, I1, W1))


train_X = spectrums1

train_Y = spectrums1

plt.plot(spectrums1[0])
plt.show()

encdec.summary()
encdec.load_weights('./SavedModel/EncDecModel.hdf5')

encdec.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-2), loss="mse")

# for i in range(10):
#     print("training cycle", i)
#     encdec.fit(train_X, train_Y, epochs=10, batch_size=2000, validation_split=0.2, shuffle=True)
#     encdec.save_weights('./SavedModel/EncDecModel.hdf5')

pred_Y = encdec(train_X)
print(pred_Y.shape, train_Y.shape)
for i in range(samples):
    plt.plot(pred_Y[i], "o")
    plt.plot(train_Y[i], "-")
    plt.show()