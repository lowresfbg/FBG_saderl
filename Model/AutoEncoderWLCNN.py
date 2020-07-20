import tensorflow as tf

def Encoder(fbg_count):
    spectra_input = tf.keras.Input((1000,))
    x = tf.expand_dims(spectra_input, axis=2)
    x = tf.keras.layers.Conv1D(fbg_count*20, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(fbg_count*20, 5, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(fbg_count*5, 5, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Permute((2,1))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)
    x = tf.keras.layers.Flatten()(x)
    return tf.keras.Model(spectra_input, x)

def Decoder(fbg_count):
    represent_input = tf.keras.Input((fbg_count*5,))
    x = tf.expand_dims(represent_input, axis=2)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(125))(x)
    x = tf.keras.layers.Permute((2,1))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(125))(x)

    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(fbg_count*20, 5, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(fbg_count*20, 5, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(1, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)

    return tf.keras.Model(represent_input, x)




def GetModel(fbg_count):
    spectra_input = tf.keras.Input((1000,))

    encoder = Encoder(fbg_count)
    decoder = Decoder(fbg_count)
    
    x = spectra_input
    
    encoded = encoder(x)
    decoded = decoder(encoded)


    wl = tf.keras.layers.Dense(fbg_count)(encoded)

    encdec = tf.keras.Model(spectra_input, decoded)
    wlcnn = tf.keras.Model(spectra_input, wl)
    return encdec, wlcnn
