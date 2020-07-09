import numpy as np
import matplotlib.pyplot as plt

from Dataset.loader import DATASET_5fbg_1 as Dataset


data = Dataset()[28]



import tensorflow as tf


# plt.plot(tf.signal.fft(tf.cast(data[1], tf.dtypes.complex64)))
# plt.yscale('log')


x = np.linspace(0,1,2001)
g = np.exp(-(x)**2/10)
# plt.plot(g)
ft = tf.signal.fft(tf.cast(g, tf.dtypes.complex64))

# plt.figure(figsize=(10,10))
plt.plot(tf.signal.ifft(tf.signal.fft(tf.cast(data[1], tf.dtypes.complex64))))
sig = tf.signal.ifft(tf.signal.fft(tf.cast(data[1], tf.dtypes.complex64)) / ft)
plt.plot(tf.math.sqrt(tf.math.imag(sig)**2+tf.math.real(sig)**2))
plt.show()

