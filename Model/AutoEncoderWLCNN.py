import tensorflow as tf

def Encoder():
    spectra_input = tf.keras.Input((1000,))
    x = tf.expand_dims(spectra_input, axis=2)
    x = tf.keras.layers.Conv1D(100, 5, activation='elu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(100, 5, activation='elu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(100, 5, activation='elu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    return tf.keras.Model(spectra_input, x)

def Decoder():
    represent_input = tf.keras.Input((125,100))
    x = tf.keras.layers.UpSampling1D(2)(represent_input)
    x = tf.keras.layers.Conv1D(100, 5, activation='elu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(100, 5, activation='elu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(1, 5, activation='elu', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    return tf.keras.Model(represent_input, x)

if __name__ == "__main__":
    Encoder().summary()
    Decoder().summary()

def GetModel(fbg_count):
    spectra_input = tf.keras.Input((1000,))

    encoder = Encoder()
    decoder = Decoder()
    
    x = spectra_input
    
    encoded = encoder(x)
    decoded = decoder(encoded)


    wl = tf.keras.layers.Flatten()(encoded)
    wl = tf.keras.layers.Dense(fbg_count)(wl)

    encdec = tf.keras.Model(spectra_input, decoded)
    wlcnn = tf.keras.Model(spectra_input, wl)
    return encdec, wlcnn
