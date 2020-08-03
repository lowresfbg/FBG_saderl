from Model import AutoEncoderWLCNN
from Dataset.Simulation.GaussCurve_TF import FBG_spectra
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from Dataset.loader import DATASET_5fbg_1_perfect as Dataset
from Dataset import Resampler


def normalize(spectra):
    maximum = tf.expand_dims(tf.reduce_max(spectra, axis=1), axis=1)
    minimum = tf.expand_dims(tf.reduce_min(spectra, axis=1), axis=1)
    return (spectra-minimum)/(maximum-minimum)


fbgs = 5
samples = 1000

x_coord = tf.linspace(1545.0, 1549.0, 1000)

X1 = tf.random.uniform([samples, fbgs])*3+1545.5
I1 = tf.repeat([[ 1, 0.5, 0.3, 0.2, 0.1  ]], samples, axis=0)*1e3
W1 = tf.ones([samples, fbgs]) * 0.2
spectrums1 = FBG_spectra(x_coord, X1, I1, W1) 
spectrums1 = spectrums1 + (tf.random.uniform(spectrums1.shape)-0.5) * 0.001

dataset =tf.concat([ tf.repeat(x_coord[tf.newaxis,tf.newaxis, :], samples, axis=0), tf.expand_dims(spectrums1, axis=1) ] , axis=1)


train_X = spectrums1
# train_Y = spectrums1
train_Y_wl = X1

# load from data set
LOAD_DATASET = False
if LOAD_DATASET:
    dataset, answer, peaks = Resampler.Resample(Dataset(), fbgs)
    spectrums1 = normalize(dataset[:, 1])
    x_coord = dataset[0, 0]

    maxy = np.max(spectrums1[0])

    train_X = spectrums1 / maxy

    train_Y = spectrums1
    train_Y_wl = answer

# ------------

# plt.plot(spectrums1[0])
# plt.show()


encdec, model, enc, dec = AutoEncoderWLCNN.GetModel(fbgs)
encdec.load_weights('./SavedModel/EncDecModel.hdf5')
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse")

def TrainModel(amount):
    print('training', amount, train_X, train_Y_wl)
    model.fit(train_X[:amount], (train_Y_wl[:amount]-1545.0)/4.0, epochs=500, batch_size=1000, verbose=1)
    



from Solution.DE_general import DE, spectra_diff_contrast, spectra_diff_absolute

from Dataset.Simulation.GaussCurve_TF import FBG_spectra, GaussCurve

from Algorithms.PeakFinder import FindPeaks

from Tests.GeneralTest_logger import SingleLogger

from AutoFitAnswer import Get3FBGAnswer


print(dataset)
print('loading completed')
ITERATION = 2000

de = DE()

def InsertAnswer(info, answer):
    i, data, X = info
    amount = 5
    if i<1:
        de.X = tf.concat([ tf.repeat([answer],amount, axis=0) + (tf.random.uniform([amount, fbgs])-0.5)*0.02 , de.X[amount:]], axis=0)



def giveAnswerGenerator(answer):
    return lambda info : InsertAnswer(info, answer)


I = I1[0]
W = W1[0]

def init(de, answer, giveAnswer = False):
    de.minx = 1545.0
    de.maxx = 1549.0
    de.I = I
    de.W = W
    de.NP = 50
    de.CR = 0.75
    de.F = 0.5
    # de.Ranged = True
    de.EarlyStop_threshold = 1e-3
    # de.spectra_diff = spectra_diff_contrast
    de.beforeEach = []

    if giveAnswer:
        de.beforeEach.append(giveAnswerGenerator(answer))




errorOverIterationsLine = []


def test(data, answer, answerTrue):
    sl = SingleLogger(I, W)
    sl.Animate = False

    init(de, answer, True)
    de.afterEach = [sl.log]

    X = de(data, max_iter=ITERATION)


    errorOverIterationsLine.append(np.sqrt(np.mean((np.array(sl.X_mean_log)-np.array(answerTrue)[np.newaxis, :])**2,axis=1)))
    


    plt.xscale('log')
    plt.yscale('log')

    for j,line in enumerate(errorOverIterationsLine):
        plt.plot(line, alpha=0.5)


    plt.tight_layout()


for i in range(0, samples, 100):

    answer = model( spectrums1[i][tf.newaxis,:] )[0]*4+1545
    answerTrue = X1[i]

    print('run ------------------')

    data = tf.concat([
        [x_coord],
        [spectrums1[i]]
    ], axis=0)
    plt.clf()

    test(data, answer, answerTrue)
    plt.pause(0.01)

    TrainModel((i+1)*100)


plt.show()