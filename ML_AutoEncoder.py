from Model import AutoEncoderWLCNN
from Dataset.Simulation.GaussCurve_TF import FBG_spectra
import tensorflow as tf
import matplotlib.pyplot as plt

fbgs = 20
samples = 40000

encdec, model = AutoEncoderWLCNN.GetModel(3)



def normalize(spectra):
    maximum = tf.expand_dims(tf.reduce_max(spectra, axis=1), axis=1)
    minimum = tf.expand_dims(tf.reduce_min(spectra, axis=1), axis=1)
    return (spectra-minimum)/(maximum-minimum)


x_coord = tf.linspace(0.0, 1.0, 1000)

X1 = tf.random.uniform([samples, fbgs])
I1 = tf.random.uniform([samples, fbgs])*2e3
W1 = tf.ones([samples, fbgs]) * tf.random.uniform([1], 0.02, 0.1)
spectrums1 = FBG_spectra(x_coord, X1, I1, W1)


train_X = spectrums1 + (tf.random.uniform([samples, 1000])-0.5)*1e-2

train_Y = spectrums1

plt.plot(train_X[0])
plt.plot(train_Y[0])
plt.show()

encdec.summary()
encdec.load_weights('./SavedModel/EncDecModel.hdf5')

encdec.compile(optimizer=tf.keras.optimizers.Adam(lr=5e-5), loss="mse")

for i in range(50):
    print("training cycle", i)
    encdec.fit(train_X, train_Y, epochs=10, batch_size=1000, validation_split=0.2, shuffle=True)
    encdec.save_weights('./SavedModel/EncDecModel.hdf5')

pred_Y = encdec(train_X[:10])
print(pred_Y.shape, train_Y.shape)
for i in range(10):
    plt.plot(pred_Y[i], "o")
    plt.plot(train_Y[i], "-")
    plt.show()
