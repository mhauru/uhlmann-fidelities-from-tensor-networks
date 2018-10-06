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
from tntools import datadispenser, multilineformatter
from tntools.yaml_config_parser import parse_argv
from tensors import Tensor
from ncon import ncon
from MPS.McMPS import McMPS
from MPS.UMPS import UMPS

tools_path = os.path.dirname(multilineformatter.__file__)
logging.config.fileConfig(tools_path + "/logging_default.conf")
filename = os.path.basename(__file__).replace(".py", "")

datadir = "uhlmann_compare_data"
if not os.path.exists(datadir):
    os.makedirs(datadir)

parinfo = {
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
    "chi1": {
        "default": 51
    },
    "chi2": {
        "default": 51
    },
    "h1": {
        "default": 1.00
    },
    "h2": {
        "default": 1.01
    },
    "L": {
        "default": 800
    },
    "do_exact": {
        "default": True
    },
    "do_separate": {
        "default": True
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
    datadispenser.update_default_pars("umps_groundstate", pars,
                                      algorithm="MPS")
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

    chi1 = pars["chi1"]
    chi2 = pars["chi2"]
    h1 = pars["h1"]
    h2 = pars["h2"]
    L = pars["L"]
    do_separate = pars["do_separate"]
    do_exact = pars["do_exact"]

    umps = datadispenser.get_data(
        dbname, "umps_groundstate", pars, algorithm="MPS",
        mps_chis=range(1,chi1), h_trans=h1
    )
    mcmps1 = McMPS(umps, tensors=[umps.tensor.copy()], weightss=[])
    logging.info("Correlation length 1: {}".format(
        mcmps1.umps.correlation_length()
    ))

    umps = datadispenser.get_data(
        dbname, "umps_groundstate", pars, algorithm="MPS",
        mps_chis=range(1,chi2), h_trans=h2
    )
    mcmps2 = McMPS(umps, tensors=[umps.tensor.copy()], weightss=[])
    logging.info("Correlation length 2: {}".format(
        mcmps2.umps.correlation_length()
    ))

    if do_separate:
        fids_separate = []
        ul, ur = None, None
        for i in range(L):
            fid, ul, ur = mcmps1.window_fidelity_separate(mcmps2, 0, i, return_us=True, initial_us=(ul, ur))
            fids_separate.append(fid)
            logging.info("{}, separate: {}".format(i, fid))
        fids_separate = np.array(fids_separate)
        with open("{}/fids_sep_latest_{}_{}_{}_{}_{}.p"
                  .format(datadir, chi1, chi2, h1, h2, L), "wb") as f:
            pickle.dump(fids_separate, f)

    if do_exact:
        fid0 = mcmps1.window_fidelity(mcmps2, 0, 0, log=True)
        fids_exact = mcmps1.window_fidelity(mcmps2, 0, L-1, upto=True, log=True)
        fids_exact = np.concatenate(([fid0], fids_exact))
        with open("{}/fids_exact_latest_{}_{}_{}_{}_{}.p"
                  .format(datadir, chi1, chi2, h1, h2, L), "wb") as f:
            pickle.dump(fids_exact, f)

if __name__ == "__main__":
    main()

