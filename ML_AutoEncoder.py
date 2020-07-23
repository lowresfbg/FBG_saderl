from Model import AutoEncoderWLCNN
from Dataset.Simulation.GaussCurve_TF import FBG_spectra
import tensorflow as tf
import matplotlib.pyplot as plt

fbgs = 5
samples = 40000

encdec, model, enc, dec = AutoEncoderWLCNN.GetModel(3)



def normalize(spectra):
    maximum = tf.expand_dims(tf.reduce_max(spectra, axis=1), axis=1)
    minimum = tf.expand_dims(tf.reduce_min(spectra, axis=1), axis=1)
    return (spectra-minimum)/(maximum-minimum)


x_coord = tf.linspace(0.0, 1.0, 1000)

X1 = tf.random.uniform([samples, fbgs])

# I1 = tf.random.uniform([samples, fbgs])*3e3

I1 = tf.repeat([[1,0.6,0.3,0.2,0.1]], samples, axis=0)*1e3 + tf.random.uniform([samples, fbgs])

# I1 *= tf.cast(tf.random.uniform([samples, fbgs])<0.9, tf.dtypes.float32)
W1 = tf.ones([samples, fbgs]) * tf.random.uniform([samples, 1], 0.03, 0.05) + tf.random.uniform([samples, fbgs])*0.01
# W1 = tf.ones([samples, fbgs]) * 0.05
spectrums1 = FBG_spectra(x_coord, X1, I1, W1)


train_X = spectrums1 + (tf.random.uniform([samples, 1000])-0.5)*tf.random.uniform([1])*0.1

train_Y = spectrums1

plt.plot(train_X[0])
plt.plot(train_Y[0])
plt.title("Close this window to continue training")
plt.show()

encdec.summary()
encdec.load_weights('./SavedModel/EncDecModel.hdf5')

encdec.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss="mse")


def plot(i):
    i%=len(train_X)
    pred_Y = encdec(train_X[i:i+1])
    codes = enc(train_X[i:i+1])
    # print(pred_Y.shape, train_Y.shape)

    plt.subplot(211)
    plt.plot(train_X[i], "-", label='input')
    plt.plot(train_Y[i], "-", label='target')
    plt.plot(pred_Y[0], "-", label='reconstructed')
    plt.legend()
    plt.subplot(212)
    plt.imshow(tf.transpose(codes[0]), aspect='auto')
    
def log(batch, logs):
    if batch%10==0:
        plt.clf()
        plot(batch)
        plt.pause(0.01)
    if batch%10==9:
        encdec.save_weights('./SavedModel/EncDecModel.hdf5')



encdec.fit(train_X, train_Y, epochs=500, batch_size=2000, validation_split=0.2, shuffle=True,
    callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=log)])


pred_Y = encdec(train_X[:10])
codes = enc(train_X[:10])
print(pred_Y.shape, train_Y.shape)

for i in range(10):
    plt.subplot(211)
    plt.plot(train_X[i], "-", label='input')
    plt.plot(train_Y[i], "-", label='target')
    plt.plot(pred_Y[i], "-", label='reconstructed')
    plt.legend()
    plt.subplot(212)
    plt.imshow(tf.transpose(codes[i]), aspect='auto')
    
    plt.show()
