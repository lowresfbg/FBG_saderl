import tensorflow as tf

def GetModel(fbg_count):
    spectra_input = tf.keras.Input((1000,))
    x = tf.expand_dims(spectra_input, axis=2)
    x = tf.keras.layers.Conv1D(fbg_count*20, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(fbg_count*20, 5, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(fbg_count*20, 5, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(fbg_count*50,activation='sigmoid')(x)
    x = tf.keras.layers.Dense(fbg_count)(x)
    return tf.keras.Model(spectra_input, x)