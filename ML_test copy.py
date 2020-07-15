from Model import SignalError
from Dataset.Simulation.GaussCurve_TF import FBG_spectra
import tensorflow as tf
import matplotlib.pyplot as plt
e_model, f_model = SignalError.ErrorModel()

samples = 100
fbgs = 2

def normalize(spectra):
    maximum = tf.expand_dims(tf.reduce_max(spectra, axis=1), axis=1)
    minimum = tf.expand_dims(tf.reduce_min(spectra, axis=1), axis=1)
    return (spectra-minimum)/(maximum-minimum)

x_coord = tf.linspace(0.0, 1.0, 1000)

# X1 = tf.random.uniform([samples, fbgs])
# I1 = tf.random.uniform([samples, fbgs], 0.1, 1)
# W1 = tf.random.uniform([samples, fbgs], 0.01, 0.2)
# spectrums1 = normalize(FBG_spectra(x_coord, X1, I1, W1))

# X2 = tf.random.uniform([samples, fbgs])
# I2 = tf.random.uniform([samples, fbgs], 0.1, 1)
# W2 = tf.random.uniform([samples, fbgs], 0.01, 0.2)
# spectrums2 = normalize(FBG_spectra(x_coord, X2, I1, W2))

goal_X = tf.constant([0.4,0.6], dtype= tf.dtypes.float32)
wrong_X = tf.constant([0.6,0.4], dtype= tf.dtypes.float32)

X1 = tf.repeat([goal_X], samples, axis=0)
I1 =  tf.repeat([[1,0.5]], samples, axis=0)
W1 = tf.ones([samples, fbgs])*0.1
spectrums1 = normalize(FBG_spectra(x_coord, X1, I1, W1))

X2 = tf.expand_dims(tf.linspace(-2.0,4.0,samples),axis=1)*(wrong_X-goal_X)+goal_X
spectrums2 = normalize(FBG_spectra(x_coord, X2, I1, W1))

train_X = tf.concat([tf.expand_dims(spectrums1, axis=1),
                     tf.expand_dims(spectrums2, axis=1)], axis=1)
                
train_Y = tf.reduce_mean(tf.abs(X2-X1), axis=1)



plt.plot(spectrums1[0])
plt.plot(spectrums2[0])
plt.show()

e_model.summary()
e_model.load_weights('./SavedModel/SignalErrorModel.hdf5')

# e_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse")

# for i in range(100):
#     print("training cycle", i)
#     e_model.fit(train_X, train_Y, epochs = 10, batch_size=1000, shuffle=True)
#     e_model.save_weights('./SavedModel/SignalErrorModel.hdf5')

pred_Y = e_model(train_X)[:,0]
print(pred_Y.shape, train_Y.shape)
plt.plot(train_Y, "o")
plt.plot(pred_Y)
plt.show()
