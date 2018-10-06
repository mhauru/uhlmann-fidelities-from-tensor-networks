import numpy as np
import scipy.integrate as integrate
import pickle
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

max_t = 20.0
max_x = 25
complexion_timestep = 0.2
insertions = "zz"
max_x_plot = 25

datafilename = (
    "quench_magnetization_data/{}expectations_{}_{}_{}.npy"
    .format(insertions, complexion_timestep, max_t, max_x)
)
max_t_step = int(np.ceil(max_t/complexion_timestep))
t_steps = list(range(max_t_step+1))
ts = [complexion_timestep*t_step for t_step in t_steps]
poses = list(range(-max_x, max_x+1))

expectations = np.load(datafilename)
max_t_step = expectations.shape[1]
t_steps = list(range(max_t_step))
ts = np.array([complexion_timestep*t_step for t_step in t_steps])

fig = plt.figure(figsize=(8,4))

# Plot a heat map over both x and t
imag_norm = np.linalg.norm(np.imag(expectations))/np.linalg.norm(expectations)
if imag_norm < 1e-10:
    print("Taking the real part, since imags are zero.")
    expectations = np.real(expectations)
else:
    print("Taking the abs, since imags are not zero.")
    expectations = np.abs(expectations)
ax = fig.add_subplot(111)
X = np.array([poses for t in ts]).transpose()
Y = np.array([ts for pos in poses])
Z = expectations

if max_x > max_x_plot:
    X = X[max_x-max_x_plot:-max_x+max_x_plot,:]
    Y = Y[max_x-max_x_plot:-max_x+max_x_plot,:]
    Z = Z[max_x-max_x_plot:-max_x+max_x_plot,:]

im = ax.pcolormesh(X, Y, Z)
ax.set_xlabel("Position $x$")
ax.set_ylabel("Time $t$")
fig.colorbar(im, orientation="horizontal")
# The same as above, but compared to the exact solution.

plt.tight_layout()
plt.savefig("quench_magnetization.pdf")
if not noshow:
    plt.show()

