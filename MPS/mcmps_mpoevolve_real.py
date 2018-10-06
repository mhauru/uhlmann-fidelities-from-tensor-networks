import numpy as np
import logging
from ncon import ncon

def mpoevolve(mcmps, mpo, pars):
    mcmps.absorb_mpo(mpo)
    error = mcmps.canonicalize(chis=pars["mps_chis"], eps=pars["mps_eps"],
                               crude=False)
    N = mcmps.length()
    eval_point = 0
    w = mcmps.weights(eval_point)
    ent = entanglement_entropy(w)
    logging.info("Length: {}".format(N))
    logging.info("Truncation error: {:.3e}".format(error))
    logging.info("Norm factors: {:.9e} & {:.9e}"
                 .format(mcmps.normfactor, mcmps.umps.normfactor))
    logging.info("Entropy at {}: {}".format(eval_point, ent))
    logging.info("Spectrum at {}:".format(eval_point))
    logging.info(w)
    return mcmps


def entanglement_entropy(w):
    w_sq = (w**2).to_ndarray()
    w_sq /= np.sum(w_sq)
    ent = -np.sum(w_sq*np.log2(w_sq))# / np.log(len(w))
    return ent


def get_operator_insertion(opname):
    if opname == "x":
        op = np.array([[ 0, 1],
                       [ 1, 0]], dtype=np.complex_)
    elif opname == "z":
        op = np.array([[ 1, 0],
                       [ 0,-1]], dtype=np.complex_)
    elif opname == "id":
        op = np.array([[ 1, 0],
                       [ 0, 1]], dtype=np.complex_)
    return op

