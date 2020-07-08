from Dataset.Simulation.GaussCurve_TF import GaussCurve, FBG_spectra
import numpy as np
import matplotlib.pyplot as plt

# print(GaussCurve(np.linspace(0,1,100), 1, 0.5,0.1).numpy())

spectra = FBG_spectra(
    np.linspace(0, 1, 100),
    np.array([[0.3, 0.7]]),
    np.array([1, 0.5]),
    0.1).numpy()

print(spectra)

plt.plot(spectra[0])
plt.show()