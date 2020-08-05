from cycler import cycler
default_cycler = (cycler(color=[
    '#3f51b5',
    '#ff5722',
    '#4caf50',
    '#e91e63',
    '#9c27b0',
    '#2196f3',
    '#fbc02d']))
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = default_cycler

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(3.89*4/3, 3.98), dpi=150)
plt.yscale('log')
axs[0,0].plot([1545,1549],[1e-10,1e-3], label="fbg")
axs[0,0].plot([1545,1549],[1e-10,1e-4], label="fbg")
axs[0,0].plot([1545,1549],[1e-10,1e-5], label="fbg")
axs[0,0].plot([1545,1549],[1e-10,1e-6], label="fbg")
axs[0,0].set_xlabel(r"$\bf{(c1)}$")
axs[0,0].set_ylabel('intensity')
axs[0,0].legend()
axs[0,0].grid()
axs[0,1].plot([1545,1549],[1e-10,1e-5], label="fbg")
axs[0,1].set_xlabel(r"$\bf{(c2)}$")
axs[0,1].legend()
axs[0,1].grid()
axs[1,0].plot([1545,1549],[1e-10,1e-7], label="fbg")
axs[1,0].set_xlabel('wavelength\n'+ r"$\bf{(c3)}$")
axs[1,0].set_ylabel('intensity')
axs[1,0].legend()
axs[1,0].grid()
axs[1,1].plot([1545,1549],[1e-10,1e-9], label="fbg")
axs[1,1].set_xlabel('wavelength\n'+ r"$\bf{(c4)}$")
axs[1,1].legend()
axs[1,1].grid()
fig.tight_layout()
plt.show()
