import tensorflow as tf

def Encoder():
    spectra_input = tf.keras.Input((1000,))
    x = tf.expand_dims(spectra_input, axis=2)

    x = tf.keras.layers.Conv1D(8, 3, activation='elu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(8, 5, activation='elu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(8, 5, activation='elu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(8, 3, activation='elu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(5)(x)
    
    return tf.keras.Model(spectra_input, x)

def Decoder():
    represent_input = tf.keras.Input((25,8))
    x = represent_input


    # # x = tf.keras.layers.Conv1D(1, 1, activation='elu', padding='same')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.UpSampling1D(5)(x)
    x = tf.keras.layers.Conv1D(8, 3, activation='elu', padding='same', 
        kernel_regularizer=tf.keras.regularizers.l1(1e-4))(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(8, 5, activation='elu', padding='same', 
        kernel_regularizer=tf.keras.regularizers.l1(1e-4))(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(8, 5, activation='elu', padding='same', 
        kernel_regularizer=tf.keras.regularizers.l1(1e-4))(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(1, 3, activation='linear', padding='same', 
        kernel_regularizer=tf.keras.regularizers.l1(1e-4))(x)
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
    
    x = encoder(x)
    decoded = decoder(x)


    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(fbg_count*10, activation='elu')(x)
    x = tf.keras.layers.Dense(fbg_count*5, activation='elu')(x)
    wl = tf.keras.layers.Dense(fbg_count)(x)

    encdec = tf.keras.Model(spectra_input, decoded)
    wlcnn = tf.keras.Model(spectra_input, wl)
    return encdec, wlcnn, encoder, decoded
