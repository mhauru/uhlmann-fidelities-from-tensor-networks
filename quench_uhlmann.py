import numpy as np
import pickle
import datetime
import logging
import logging.config
import os
import sys
import datetime
import configparser
import warnings
from MPS.mcmps_mpoevolve_real import get_operator_insertion, entanglement_entropy
from tntools import datadispenser, multilineformatter
from tntools.yaml_config_parser import parse_argv
from tensors import Tensor
from ncon import ncon
from MPS.McMPS import McMPS
from MPS.UMPS import UMPS

tools_path = os.path.dirname(multilineformatter.__file__)
logging.config.fileConfig(tools_path + "/logging_default.conf")
filename = os.path.basename(__file__).replace(".py", "")

datadir = "quench_uhlmann_data"
if not os.path.exists(datadir):
    os.makedirs(datadir)

parinfo = {
    "insertion": {
        "default": "z"
    },
    "max_t": {
        "default": 20
    },
    "verbosity": {
        "default": 10
    },
    "symmetry_tensors": {
        "default": False
    },
    "debug": {
        "default": False
    },
    "database": {
        "default": "./data/"
    },
    "llimit": {
        "default": -25
    },
    "rlimit": {
        "default": 25
    }
}


def parse():
    pars = parse_argv(sys.argv)
    return pars


def apply_default_pars(pars, parinfo):
    for k, v in parinfo.items():
        if k not in pars:
            pars[k] = v["default"]
    return


def set_filehandler(logger, logfilename, pars):
    os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    filehandler = logging.FileHandler(logfilename, mode='w')
    if pars["debug"]:
        filehandler.setLevel(logging.DEBUG)
    else:
        filehandler.setLevel(logging.INFO)
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(tools_path + '/logging_default.conf')
    fmt = parser.get('formatter_default', 'format')
    datefmt = parser.get('formatter_default', 'datefmt')
    formatter = multilineformatter.MultilineFormatter(fmt=fmt, datefmt=datefmt)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return


def main():
    pars = parse()
    apply_default_pars(pars, parinfo)
    datadispenser.update_default_pars("timeevolved_insertion_mcmps", pars,
                                      algorithm="MPS", t=3)
    dbname = pars["database"]
    if pars["debug"]:
        warnings.filterwarnings('error')

    datetime_str = datetime.datetime.strftime(datetime.datetime.now(),
                                             '%Y-%m-%d_%H-%M-%S')
    title_str = ('{}_{}'.format(filename, datetime_str))
    logfilename = "logs/{}.log".format(title_str)
    rootlogger = logging.getLogger()
    set_filehandler(rootlogger, logfilename, pars)

    # - Infoprint -
    infostr = "\n{}\n".format("="*70)
    infostr += "Running {} with the following parameters:".format(filename)
    for k,v in sorted(pars.items()):
        infostr += "\n%s = %s"%(k, v)
    logging.info(infostr)

    # Find the range of time and position for evaluating the
    # expectations.
    max_t_step = int(np.ceil(pars["max_t"]/pars["lorentzion_timestep"]))
    t_steps = list(range(max_t_step+1))
    ts = [pars["lorentzion_timestep"]*t_step for t_step in t_steps]

    fid_t_file = datadir + "/fid_t_latest.npy"

    llimit = pars["llimit"]
    rlimit = pars["rlimit"]
    dist = rlimit - llimit
    if os.path.exists(fid_t_file):
        os.remove(fid_t_file)
    fid_t = np.empty((dist, 3, 0), dtype=np.complex_)

    for t, t_step in zip(ts, t_steps):
        t = np.around(t, 10)
        logging.info("\nt: {}".format(t))
        mcmps = datadispenser.get_data(
            dbname, "timeevolved_insertion_mcmps", pars, t=t, algorithm="MPS"
        )
        N = mcmps.length()

        eval_point = 0
        w = mcmps.weights(eval_point)
        ent = entanglement_entropy(w)
        logging.info("Length: {}".format(N))
        logging.info("Norm factors: {:.9e} & {:.9e}"
                     .format(mcmps.normfactor, mcmps.umps.normfactor))

        umps = datadispenser.get_data(
            dbname, "umps_groundstate", pars, algorithm="MPS"
        )
        conj_mps = type(mcmps)(umps)
        # This identifier changing is some ad hoc crap that shouldn't be
        # necessary.
        conj_mps.change_identifier()
        mcmps.change_identifier()
        temp1 = np.array([mcmps.halfsystem_fidelity(conj_mps, i+1/2,
                                                    normalize=False)
                          for i in range(llimit, rlimit)])
        temp2 = np.array([mcmps.window_fidelity(conj_mps, i, i+1,
                                                normalize=False)
                          for i in range(llimit, rlimit)])
        temp2 = np.reshape(temp2, (temp2.shape[0], 1))
        temp = np.concatenate((temp1, temp2), axis=1)
        temp = np.reshape(temp, (temp.shape[0], 3, 1))
        fid_t = np.concatenate((fid_t, temp), axis=2)
        np.save(fid_t_file, fid_t)


if __name__ == "__main__":
    main()

