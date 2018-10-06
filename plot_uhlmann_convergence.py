import numpy as np
import scipy.integrate as integrate
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
plot_file = "uhlmann_convergence.pdf"

L = 300
L_plot = 300

def fid_func(fids):
    fids = 1-fids
    fids = fids[:L_plot]
    return fids

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,1,1)

with open("./{}/fids_exact_latest_21_11_1.0_1.0_{}.p".format(datadir, L), "rb") as f:
    fids = pickle.load(f)
    fids = np.array(fids)
    fids = fid_func(fids)
    ax.semilogy(fids, ls=":",
                label="$F, \; h=1.0$",
                color="blue")

with open("./{}/fids_sep_latest_21_11_1.0_1.0_{}.p".format(datadir, L), "rb") as f:
    fids = pickle.load(f)
    fids = np.array(fids)
    fids = fid_func(fids)
    ax.semilogy(fids, ls="-",
                label="$F_{\mathrm{d}}, \; h=1.0$",
                color="blue")

with open("./{}/fids_exact_latest_21_11_1.05_1.05_{}.p".format(datadir, L), "rb") as f:
    fids = pickle.load(f)
    fids = np.array(fids)
    fids = fid_func(fids)
    ax.semilogy(fids, ls=":",
                label="$F, \; h=1.05$",
                color="green")

with open("./{}/fids_sep_latest_21_11_1.05_1.05_{}.p".format(datadir, L), "rb") as f:
    fids = pickle.load(f)
    fids = np.array(fids)
    fids = fid_func(fids)
    ax.semilogy(fids, ls="-",
                label="$F_{\mathrm{d}}, \; h=1.05$",
                color="green")

ax.set_ylim(1e-10, 1e-1)
ax.set_xlabel("Window size")
ax.set_ylabel("$1 - $ fidelity")
ax.legend()

plt.savefig(plot_file)

if not noshow:
    plt.show()

