import numpy as np
import logging
from ncon import ncon
from . import umps_mpoevolve
from . import mcmps_mpoevolve_real
from .McMPS import McMPS
from .UMPS import UMPS

version = 1.0

parinfo = {
    "mps_chis": {
        "default": range(1,31),
        "idfunc": lambda dataname, pars: True
    },
    "ground_umps_chis": {
        "default": None,
        "idfunc": lambda dataname, pars: (
            dataname == "timeevolved_insertion_mcmps"
            and pars["ground_umps_chis"]
            and not tuple(pars["ground_umps_chis"]) == tuple(pars["mps_chis"])
        )
    },
    "mps_eps": {
        "default": 1e-5,
        "idfunc": lambda dataname, pars: True
    },
    "verbosity": {
        "default": np.inf,
        "idfunc": lambda dataname, pars: False
    },
    # Generating UMPS ground states.
    # TODO See umps_mpoevolve for why these are used here.
    "euclideon_spacestep": {
        "default": 1,
        "idfunc": lambda dataname, pars: dataname == "umps_groundstate"
    },
    "euclideon_padding": {
        "default": 3,
        "idfunc": lambda dataname, pars: dataname == "umps_groundstate"
    },
    "euclideon_eps": {
        "default": 1e-5,
        "idfunc": lambda dataname, pars: dataname == "umps_groundstate"
    },
    "euclideon_chis": {
        "default": range(1,7),
        "idfunc": lambda dataname, pars: dataname == "umps_groundstate"
    },
    "initial_timestep": {
        "default": 50,
        "idfunc": lambda dataname, pars: dataname == "umps_groundstate"
    },
    "min_timestep": {
        "default": 1/8,
        "idfunc": lambda dataname, pars: dataname == "umps_groundstate"
    },
    "timestep_decreasant": {
        "default": 0.8,
        "idfunc": lambda dataname, pars: dataname == "umps_groundstate"
    },
    "max_counter": {
        "default": 1000,
        "idfunc": lambda dataname, pars: dataname == "umps_groundstate"
    },
    "max_subcounter": {
        "default": 15,
        "idfunc": lambda dataname, pars: dataname == "umps_groundstate"
    },
    "entropy_eps": {
        "default": 1e-3,
        "idfunc": lambda dataname, pars: dataname == "umps_groundstate"
    },
    "energy_eps": {
        "default": 1e-3,
        "idfunc": lambda dataname, pars: dataname == "umps_groundstate"
    },
    "spectrum_eps": {
        "default": np.inf,
        "idfunc": lambda dataname, pars: (
            dataname == "umps_groundstate" and pars["spectrum_eps"] < np.inf
        )
    },
    # MPO time evolution
    "lorentzion_timestep": {
        "default": 0.2,
        "idfunc": lambda dataname, pars: dataname == "timeevolved_insertion_mcmps"
    },
    "lorentzion_spacestep": {
        "default": 1,
        "idfunc": lambda dataname, pars: dataname == "timeevolved_insertion_mcmps"
    },
    "lorentzion_padding": {
        "default": 3,
        "idfunc": lambda dataname, pars: dataname == "timeevolved_insertion_mcmps"
    },
    "lorentzion_eps": {
        "default": 1e-5,
        "idfunc": lambda dataname, pars: dataname == "timeevolved_insertion_mcmps"
    },
    "lorentzion_chis": {
        "default": range(1,7),
        "idfunc": lambda dataname, pars: dataname == "timeevolved_insertion_mcmps"
    },
    "insertion_early": {
        "default": "z",
        "idfunc": lambda dataname, pars: dataname == "timeevolved_insertion_mcmps"
    },
    "t": {
        "default": 0,
        "idfunc": lambda dataname, pars: dataname == "timeevolved_insertion_mcmps"
    },
    "timesmear_sigma": {
        "default": 0,
        "idfunc": lambda dataname, pars: (
            dataname == "timeevolved_insertion_mcmps"
            and pars["timesmear_sigma"] != 0
        )
    },
    "timesmear_euclideon_timestep": {
        "default": 1e-3,
        "idfunc": lambda dataname, pars: (
            dataname == "timeevolved_insertion_mcmps"
            and pars["timesmear_sigma"] != 0
        )
    },
    "timesmear_euclideon_spacestep": {
        "default": 1,
        "idfunc": lambda dataname, pars: (
            dataname == "timeevolved_insertion_mcmps"
            and pars["timesmear_sigma"] != 0
        )
    },
    "timesmear_euclideon_padding": {
        "default": 3,
        "idfunc": lambda dataname, pars: (
            dataname == "timeevolved_insertion_mcmps"
            and pars["timesmear_sigma"] != 0
        )
    },
    "timesmear_euclideon_eps": {
        "default": 1e-10,
        "idfunc": lambda dataname, pars: (
            dataname == "timeevolved_insertion_mcmps"
            and pars["timesmear_sigma"] != 0
        )
    },
    "timesmear_euclideon_chis": {
        "default": range(1,7),
        "idfunc": lambda dataname, pars: (
            dataname == "timeevolved_insertion_mcmps"
            and pars["timesmear_sigma"] != 0
        )
    },
}


def generate(dataname, *args, pars=dict(), filelogger=None):
    infostr = ("{}"
               "\nGenerating {} with MPS version {}."
               .format("="*70, dataname, version))
    logging.info(infostr)
    if filelogger is not None:
        # Only print the dictionary into the log file, not in stdout.
        dictstr = ""
        for k,v in sorted(pars.items()):
           dictstr += "\n%s = %s"%(k, v)
        filelogger.info(dictstr)

    if dataname == "umps_groundstate":
        res = generate_umps_groundstate(*args, pars=pars)
    elif dataname == "timeevolved_insertion_mcmps":
        res = generate_timeevolved_insertion_mcmps(*args, pars=pars)
    else:
        raise ValueError("Unknown dataname: {}".format(dataname))
    return res


def generate_umps_groundstate(*args, pars=dict()):
    ham = args[0]
    res = umps_mpoevolve.optimize_groundstate(ham, pars)
    return res


def generate_timeevolved_insertion_mcmps(*args, pars=dict()):
    mcmps = args[0]
    if type(mcmps) != McMPS:
        umps = mcmps
        # It's presumably an UMPS, in need of an operator insertion.
        # Create a McMPS out of punching an UMPS with op.
        op = mcmps_mpoevolve_real.get_operator_insertion(pars["insertion_early"])
        op = umps.tensortype().from_ndarray(op)
        tensor = umps.tensor
        weights = umps.weights
        tensor_op = ncon((tensor, op), ([-1,2,-3], [2,-2]))
        mcmps = McMPS(umps, tensors=[tensor_op], weightss=[])
        if pars["timesmear_sigma"] > 0:
            A = args[2][0]
            smear_t = 0
            while smear_t < pars["timesmear_sigma"]**2/2:
                logging.info("Smearing 'time': {}".format(smear_t))
                mcmps = mcmps_mpoevolve_real.mpoevolve(mcmps, A, pars)
                smear_t += pars["timesmear_euclideon_timestep"]
                smear_t = np.around(smear_t, 10)
            mcmps.normalize()
            mcmps.normfactor /= np.sqrt(2*np.pi)*pars["timesmear_sigma"]
    A = args[1][0]
    if np.abs(pars["t"]) < 1e-12:
        # We are essentially at time zero, no need to time evolve.
        return mcmps
    res = mcmps_mpoevolve_real.mpoevolve(mcmps, A, pars)
    return res


def prereq_pairs(dataname, pars):
    if dataname == "umps_groundstate":
        res = prereq_pairs_umps_groundstate(pars)
    elif dataname == "timeevolved_insertion_mcmps":
        res = prereq_pairs_timeevolved_insertion_mcmps(pars)
    else:
        raise ValueError("Unknown dataname: {}".format(dataname))
    return res


def prereq_pairs_umps_groundstate(pars):
    prereq_pars = pars.copy()
    res = [("ham", prereq_pars)]
    return res


def prereq_pairs_timeevolved_insertion_mcmps(pars):
    prereq_pars = pars.copy()
    prereq_pars["t"] -= prereq_pars["lorentzion_timestep"]
    prereq_pars["t"] = np.around(prereq_pars["t"], 10)

    if prereq_pars["t"] < 0:
        dataname = "umps_groundstate"
        if pars["ground_umps_chis"]:
            prereq_pars["mps_chis"] = pars["ground_umps_chis"]
    else:
        dataname = "timeevolved_insertion_mcmps"

    complexion_pars = pars.copy()
    complexion_pars["complexion_step_direction"] = 1j
    complexion_pars["complexion_timestep"] = pars["lorentzion_timestep"]
    complexion_pars["complexion_spacestep"] = pars["lorentzion_spacestep"]
    complexion_pars["complexion_chis"] = pars["lorentzion_chis"]
    complexion_pars["complexion_padding"] = pars["lorentzion_padding"]
    complexion_pars["complexion_eps"] = pars["lorentzion_eps"]
    complexion_pars["iter_count"] = 0
    complexion_pars["model"] = "complexion_" + pars["model"]
    res = ((dataname, prereq_pars), ("A", complexion_pars))
    if prereq_pars["t"] < 0 and pars["timesmear_sigma"] > 0:
        euclideon_pars = pars.copy()
        euclideon_pars["complexion_step_direction"] = -1
        euclideon_pars["complexion_timestep"] = pars["timesmear_euclideon_timestep"]
        euclideon_pars["complexion_spacestep"] = pars["timesmear_euclideon_spacestep"]
        euclideon_pars["complexion_chis"] = pars["timesmear_euclideon_chis"]
        euclideon_pars["complexion_padding"] = pars["timesmear_euclideon_padding"]
        euclideon_pars["complexion_eps"] = pars["timesmear_euclideon_eps"]
        euclideon_pars["iter_count"] = 0
        euclideon_pars["model"] = "complexion_sq_" + pars["model"]
        res += (("A", euclideon_pars),)
    return res


