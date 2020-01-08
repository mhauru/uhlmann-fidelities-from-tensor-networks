In this folder you can find Python 3/NumPy implementations of the tensor
network methods described in the paper "Uhlmann fidelities from tensor
networks". This code can be used to reproduce the benchmark results for the
Ising model, shown in the paper. It should not be considered a reference
implementation, as it is simply a semi-cleaned snap shot of the author's
personal codebase. It includes unused, possibly broken bits, does certain
things in unnecessarily complicated and over general ways, and is conversely
ad hoc at other times. The main purpose of the code is to function as the
ultimate reference for how the results in the paper were produced, and at
the same time as a proof that the algorithms described in the paper really
do produce the plots shown.

The main content is in the files MPS/UMPS.py and MPS/McMPS.py. UMPS.py
implements a class for uniform, infinite Matrix Product States. McMPS
implements a class for infinite Matrix Product States, with a non-uniform
window in the middle, and uniform parts at the ends. In these two files you
can find all the functions for evaluating Uhlmann fidelities, as described in
the paper. These two are also the most polished of the source code files, and
should be quite readable. Note that the core content of implementing what the
paper describes, is in a small handful of functions in these two files.
Everything else in this folder is infrastructure built to produce benchmark
results for the Ising model.

The rest of the source code divides into three categories:

1) Tensor network packages
The code makes extensive use of the `tensors`, `ncon`, and `tntools` packages,
which provide basic tensor network routines, as well as some generally useful
convenience classes and methods. These packages can also be found at
https://github.com/mhauru.

2) Plotting code
Quite self-explanatory. See the the files plot_*.py.

2) Supporting code and files
This includes for instance the files in the folder `confs`, which specify the
parameters for the various Python files. Most importantly, this also includes
the `makefile`. It is set up in such a way, that on Unix systems simply running
`make`
should generate all the benchmark results and plots thereof. Other commands
such as `make data_quench_magnetization` and `make plots_convergence` can be
used to only produce some of the data/plots. See the `makefile` for more
details.

Note that some of the data uses somewhat high bond dimensions, and the MPS
implementation provided here also isn't the most performant. Running `make`
should be possible on most modern personal computers, but may take a day or so
to finish. One can of course set up parallel jobs on a cluster to speed things
up.

The code is licensed under the MIT license, as described in the file LICENSE.

For any questions, or help with using the code, please contact Markus Hauru at
markus@mhauru.org.
