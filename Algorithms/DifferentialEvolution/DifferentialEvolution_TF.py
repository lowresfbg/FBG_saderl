import numpy as np
import tensorflow as tf

@tf.function
def Mutate(X, CR=0.5, F=1):
    NP = X.shape[0]

    i = tf.repeat([tf.range(NP)], NP, axis=0)
    i1 = tf.repeat([tf.range(NP-1)], NP, axis=0)
    i2 = tf.repeat([tf.range(NP-2)], NP, axis=0)
    i3 = tf.repeat([tf.range(NP-3)], NP, axis=0)

    a = tf.repeat(tf.expand_dims(tf.random.uniform(
        [NP], 0, NP-1, tf.dtypes.int32), axis=1), NP-1, axis=1)
    b = tf.repeat(tf.expand_dims(tf.random.uniform(
        [NP], 0, NP-2, tf.dtypes.int32), axis=1), NP-2, axis=1)
    c = tf.repeat(tf.expand_dims(tf.random.uniform(
        [NP], 0, NP-3, tf.dtypes.int32), axis=1), NP-3, axis=1)

    remain = tf.reshape(tf.boolean_mask(i, i != tf.transpose(i)), (NP, -1))
    ai = remain[i1 == a]
    remain = tf.reshape(tf.boolean_mask(remain, i1 != a), (NP, -1))
    bi = remain[i2 == b]
    remain = tf.reshape(tf.boolean_mask(remain, i2 != b), (NP, -1))
    ci = remain[i3 == c]
    # print(ai)
    # print(bi)
    # print(ci)

    V = tf.gather(X, ai) + F*(tf.gather(X, bi)-tf.gather(X, ci))

    X = X + (V-X)*tf.cast(tf.random.uniform(X.shape) < CR, tf.dtypes.float32)
    return X


# print(Mutate(np.array([0, 1, 2, 3])))
