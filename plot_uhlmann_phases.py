import numpy as np
import pickle
import warnings
import datetime
import sys
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from ncon import ncon
from tntools import datadispenser

noshow = "noshow" in sys.argv
practice = "practice" in sys.argv

datadir = "uhlmann_compare_data"
plot_file = "uhlmann_phases.pdf"

L = 800
chi = 21 if practice else 51
h1 = 1.0
h2 = 1.01

L_plot = 800
correlation_length = 92.53254929694677  # Read from the log file manually.

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,1,1)
ax_diff = ax.twinx()

if correlation_length:
    ax.axvline(x=correlation_length, ls="-", color="darkgrey")

with open("./{}/fids_exact_latest_{}_{}_{}_{}_{}.p".format(datadir, chi, chi, h1, h2, L), "rb") as f:
    fids = pickle.load(f)
    fids = np.array(fids)
    ax.plot(fids[:L_plot], ls=":", label="$F$", color="blue")
    ax_diff.plot(np.diff(fids)[:L_plot], ls=":", color="green")

with open("./{}/fids_sep_latest_{}_{}_{}_{}_{}.p".format(datadir, chi, chi, h1, h2, L), "rb") as f:
    fids = pickle.load(f)
    fids = np.array(fids)
    ax.plot(fids[:L_plot], ls="-", label="$F_{d}$", color="blue")
    ax_diff.plot(np.diff(fids)[:L_plot], ls="-", color="green")

ax.set_ylim(0.65, 1.02)
ax_diff.set_ylim(-0.001, 0)
ax.set_xlabel("Window size")
ax.set_ylabel("Fidelity")
ax_diff.set_ylabel("Derivative of fidelity")
ax.legend()
fig.tight_layout()

plt.savefig(plot_file)

if not noshow:
    plt.show()

