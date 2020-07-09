import numpy as np
import matplotlib.pyplot as plt


# from Dataset.Simulation.GaussCurve_TF import GaussCurve, FBG_spectra

# print(GaussCurve(np.linspace(0,1,100), 1, 0.5,0.1).numpy())

# spectra = FBG_spectra(
#     np.linspace(0, 1, 100),
#     np.array([[0.3, 0.7]]),
#     np.array([1, 0.5]),
#     0.1).numpy()

# print(spectra)

# plt.plot(spectra[0])
# plt.show()

"""
--------------------
"""


from Model.SpectrumCompensate import SpectrumCompensateModel
import tensorflow as tf

SM, SCM = SpectrumCompensateModel.GetModel()

SCM.compile(optimizer="adam", loss="mse")

result = SCM([
    tf.constant([np.linspace(0, 1, 100)]),
    tf.constant([[[1, 0.3, 0.2], [0.5, 0.7, 0.4]]])
])


plt.plot(result[0])
plt.show()