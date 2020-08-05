import numpy as np
from AutoFitAnswer import GetFBGAnswer

from scipy.interpolate import interp1d
import tensorflow as tf
from Algorithms.PeakFinder import FindPeaks

def Resample(dataset, fbg_count, skip=1, target=1000, threshold=1.5e-5):

    if type(dataset) is tuple:
        print('is tuple!')
        dataset, answertable = dataset
        # print(len(dataset), answertable.shape)
        answer = answertable[:, :fbg_count]
        peaks = tf.constant(np.concatenate([
            answertable[0, fbg_count:fbg_count*2, np.newaxis],
            answertable[0, :fbg_count, np.newaxis],
            answertable[0, fbg_count*2:, np.newaxis],
        ], axis=1), dtype=tf.dtypes.float32)
    else:
        answer = np.array(GetFBGAnswer(dataset, 3, threshold)).T
        peaks = tf.constant(
            FindPeaks(dataset[0], threshold), dtype=tf.dtypes.float32)

    dataset = Sample(dataset, target, skip)

    return dataset, answer, peaks

def Sample(dataset, target, skip=1):
    dataset = tf.constant(dataset, dtype=tf.dtypes.float32)[:, :, ::skip]
    print(dataset)
    x_coord = np.linspace(np.min(dataset[:, 0]), np.max(dataset[:, 0]), target)
    new_dataset = []
    for i in range(dataset.shape[0]):
        new_dataset.append(
            interp1d(dataset[i, 0], dataset[i, 1],  kind='cubic')(x_coord))
    dataset = tf.cast(tf.concat([
        tf.repeat([[x_coord]], dataset.shape[0], axis=0),
        np.expand_dims(new_dataset, axis=1)], axis=1), tf.dtypes.float32)
    return dataset
