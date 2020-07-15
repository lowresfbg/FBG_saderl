import tensorflow as tf


def FeatureModel():
    spectra_input = tf.keras.Input((1000,))
    x = tf.expand_dims(spectra_input, axis=2)
    x = tf.keras.layers.Conv1D(50, 10, activation='sigmoid')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(20, 5, activation='sigmoid')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(10, 3, activation='sigmoid')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    return tf.keras.Model(spectra_input, x)

def ErrorModel():
    feature_model = FeatureModel()

    spectra_input = tf.keras.Input((2,1000))
    feature1 = feature_model(spectra_input[:,0])
    feature2 = feature_model(spectra_input[:,1])
    x = tf.abs(feature1-feature2)
    # x = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(spectra_input, x), feature_model


