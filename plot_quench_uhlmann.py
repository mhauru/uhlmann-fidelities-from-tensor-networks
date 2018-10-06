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

llimit = -25
rlimit = 25
plot_llimit = -25
plot_rlimit = 25
t_step = 0.2
ts = (0, 5, 10, 20)
ms = 3

fid_t_file = "./quench_uhlmann_data/fid_t_latest.npy"
plot_file = "quench_uhlmann.pdf"

l_index = plot_llimit - llimit
r_index = plot_rlimit - llimit

poses = np.array([i+1/2 for i in range(llimit, rlimit)])
poses = poses[l_index:r_index]

fid_t = np.load(fid_t_file)

fig = plt.figure(figsize=(8,4))
plots = len(ts)
rows = 1
columns = 1
while rows*columns < plots:
    if columns > 1.5*rows:
        rows += 1
    else:
        columns += 1

for i, t in enumerate(ts):
    t_index = int(np.round(t/t_step))
    fid_l = fid_t[:, 0, t_index]
    fid_r = fid_t[:, 1, t_index]

    ax = fig.add_subplot(columns, rows, i+1)
    if fid_t.shape[1] > 2:
        fid_twosite = fid_t[:, 2, t_index]
        ax.plot(poses, np.abs(fid_twosite[l_index:r_index]),
                ls="", marker="o", ms=ms, label="$F$ two-site")
    ax.plot(poses, np.abs(fid_l[l_index:r_index]),
            ls="", marker="o", ms=ms, label="$F$ left")
    ax.plot(poses, np.abs(fid_r[l_index:r_index]),
            ls="", marker="o", ms=ms, label="$F$ right")
    ax.set_ylim(0.55, 1.05)
    if i % 2 == 0:
        ax.set_ylabel("Fidelity $F$")
    if i >= len(ts)/2:
        ax.set_xlabel("Position $x$")
    ax.set_title("$t = {}$".format(t))
    if i == 0:
        ax.legend()

fig.tight_layout()

plt.savefig(plot_file)

if not noshow:
    plt.show()

