import numpy as np
import pickle
import datetime
import logging
import logging.config
import os
import sys
import datetime
import configparser
from tntools import datadispenser, multilineformatter
from tntools.yaml_config_parser import parse_argv
from tensors import Tensor
from ncon import ncon
from MPS.McMPS import McMPS
from MPS.UMPS import UMPS
from MPS.mcmps_mpoevolve_real import get_operator_insertion, entanglement_entropy

tools_path = os.path.dirname(multilineformatter.__file__)
logging.config.fileConfig(tools_path + "/logging_default.conf")
filename = os.path.basename(__file__).replace(".py", "")

datadir = "quench_magnetization_data"
if not os.path.exists(datadir):
    os.makedirs(datadir)

parinfo = {
    "insertion_early": {
        "default": "z"
    },
    "insertion_late": {
        "default": "z"
    },
    "max_t": {
        "default": 20
    },
    "max_x": {
        "default": 25
    },
    "verbosity": {
        "default": 10
    },
    "debug": {
        "default": False
    },
    "database": {
        "default": "./data/"
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


def update_expectations(mcmps, op, t, expectations, poses, pars):
    data_filename = (
        "{}/{}{}expectations_{}_{}_{}.npy"
        .format(datadir, pars["insertion_early"], pars["insertion_late"],
                pars["lorentzion_timestep"], t, pars["max_x"])
    )
    logging.info("Evaluating expectations.")
    values = [mcmps.expect_local(op, i, normalize=False) for i in poses]
    values = np.array(values)
    values = np.reshape(values, (values.shape[0], 1))
    if expectations is not None:
        expectations = np.concatenate((expectations, values), axis=1)
    else:
        expectations = values
    np.save(data_filename, expectations)
    logging.info("Done.")
    return expectations, data_filename


def main():
    pars = parse()
    apply_default_pars(pars, parinfo)
    datadispenser.update_default_pars("timeevolved_insertion_mcmps", pars,
                                      algorithm="MPS", t=5)
    datadispenser.update_default_pars(
        "A", pars, iter_count=0, complexion_step_direction=-1,
        algorithm="TNR", model="complexion_"+pars["model"],
        complexion_spacestep=1, complexion_timestep=1
    )
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
    poses = list(range(-pars["max_x"], pars["max_x"]+1))
    max_t_step = int(np.ceil(pars["max_t"]/pars["lorentzion_timestep"]))
    t_steps = list(range(max_t_step+1))
    ts = [pars["lorentzion_timestep"]*t_step for t_step in t_steps]

    mcmps = datadispenser.get_data(
        dbname, "timeevolved_insertion_mcmps", pars, t=0,
        algorithm="MPS"
    )
    if mcmps.tensortype() == Tensor:
        pars["symmetry_tensors"] = False
    else:
        pars["symmetry_tensors"] = True
    op_late = get_operator_insertion(pars["insertion_late"])
    op_late = mcmps.tensortype().from_ndarray(op_late)

    # Evaluate correlators for various times.
    data_filename = None
    expectations = None
    for t, t_step in zip(ts, t_steps):
        t = np.around(t, 10)
        logging.info("\nt: {}".format(t))
        mcmps = datadispenser.get_data(
            dbname, "timeevolved_insertion_mcmps", pars, t=t,
            algorithm="MPS"
        )
        N = mcmps.length()

        eval_point = 0
        w = mcmps.weights(eval_point)
        ent = entanglement_entropy(w)
        logging.info("Length: {}".format(N))
        logging.info("Norm factors: {:.9e} & {:.9e}"
                     .format(mcmps.normfactor, mcmps.umps.normfactor))
        logging.info("Entropy at {}: {}".format(eval_point, ent))
        logging.info("Spectrum at {}:".format(eval_point))
        logging.info(w)

        old_data_filename = data_filename
        expectations, data_filename, = update_expectations(
            mcmps, op_late, t, expectations, poses, pars
        )
        if old_data_filename is not None:
            os.remove(old_data_filename)


if __name__ == "__main__":
    main()
