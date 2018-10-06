import numpy as np
import warnings
import logging
from tntools import initialtensors
from ncon import ncon
from .UMPS import UMPS
from .McMPS import McMPS

def mpoevolve(umps, mpo, pars):
    umps.absorb_mpo(mpo)
    res = umps.canonicalize(chis=pars["mps_chis"], eps=pars["mps_eps"],
                            crude=True, confirm=False)
    error = res[-1]
    logging.info("Truncation error: {}".format(error))
    return umps

    
def entanglement_entropy(w):
    w_sq = (w**2).to_ndarray()
    w_sq /= np.sum(w_sq)
    ent = -(w_sq*np.log2(w_sq)).sum()# / np.log2(len(w))
    return ent


def optimize_groundstate(ham, pars):
    phys_dim = ham.shape[0]
    if ham.qhape is not None:
        virt_dim = [1]*len(ham.qhape[0])
    else:
        virt_dim = max(1, min(pars["mps_chis"]))
    umps = UMPS.random(phys_dim, virt_dim, tensorcls=type(ham))

    energy = np.real(umps.expect_local(ham))
    ent = entanglement_entropy(umps.weights)
    counter = 0
    sub_counter = 0
    step = pars["initial_timestep"]
    next_step = step*pars["timestep_decreasant"]

    logging.info("Virtual dimension: {}".format(umps.virtualdim()))
    logging.info("Correlation length: {}".format(umps.correlation_length()))
    logging.info("Energy: {}".format(energy))
    logging.info("Entropy: {}".format(ent))

    # TODO Not great, that we are calling the internals of
    # initialtensors. This is to avoid getting huge numbers of
    # prerequisites from datadispenser, but a better solution would be
    # better.
    A = initialtensors.build_complexion(
        ham, pars, complexion_step_direction=-1,
        complexion_timestep=step,
        complexion_spacestep=pars["euclideon_spacestep"],
        complexion_padding=pars["euclideon_padding"],
        complexion_eps=pars["euclideon_eps"],
        complexion_chis=pars["euclideon_chis"]
    )

    while counter < pars["max_counter"]:
        sub_counter += 1
        counter += 1
        logging.info("\nCounter: {} ({})".format(counter, sub_counter))
        logging.info("Time step: {}".format(step))
        old_spectrum = umps.weights
        umps_old = umps.copy()
        mpoevolve(umps, A, pars)
        # Euclidean evolution isn't even supposed to conserve the norm.
        umps.reset_normfactor()

        old_energy = energy
        energy = np.real(umps.expect_local(ham))
        energy_change = np.abs((energy - old_energy)/energy)
        old_ent = ent
        ent = entanglement_entropy(umps.weights)
        ent_change = np.abs((ent - old_ent)/ent)
        spectrum = umps.weights
        try:
            diff = spectrum - old_spectrum
        except (AssertionError, ValueError):
            diff = spectrum
        spectrum_change = diff.norm()

        # Dominant eigenvalue of mixed transfer matrix.
        from tntools.ncon_sparseeig import ncon_sparseeig
        A_umps = umps.get_rightweight_tensor()
        A_umps_old_conj = umps_old.get_rightweight_tensor().conjugate()
        S, U = ncon_sparseeig(
            (A_umps, A_umps_old_conj), ([-11,3,-1], [-12,3,-2]),
            right_inds=[0,1], left_inds=[2,3],
            matvec_order=[1,2,3], rmatvec_order=[11,12,3],
            matmat_order=[1,2,3], chis=[1]
        )
        dom_eigval = S.max()

        # New/old fidelities
        mcmps_umps = McMPS(umps)
        mcmps_umps_old = McMPS(umps_old)
        fid = mcmps_umps.window_fidelity(mcmps_umps_old, 0, 0)
        fid_sep = mcmps_umps.window_fidelity_separate(mcmps_umps_old, 0, 0)

        logging.info("Virtual dimension: {}".format(umps.virtualdim()))
        logging.info("Correlation length: {}".format(umps.correlation_length()))
        logging.info("Energy: {} ({})".format(energy, energy_change))
        logging.info("Entropy: {} ({})".format(ent, ent_change))
        logging.info("Spectrum change: {}".format(spectrum_change))
        logging.info("1 - dominant eigenvalue of mixed transfermatrix: {}".format(1-dom_eigval))
        logging.info("1 - one-site fidelity: {}".format(1-fid))
        logging.info("1 - one-site separable fidelity: {}".format(1-fid_sep))
        
        converged = (ent_change < pars["entropy_eps"]
                     and energy_change < pars["energy_eps"]
                     and spectrum_change < pars["spectrum_eps"])
        if (converged or
                (sub_counter > pars["max_subcounter"]
                 and next_step > pars["min_timestep"])):
            step = next_step
            next_step = step*pars["timestep_decreasant"]
            sub_counter = 0
            if step < pars["min_timestep"]:
                break
            else:
                # TODO Not great, that we are calling the internals of
                # initialtensors. This is to avoid getting huge numbers of
                # prerequisites from datadispenser, but a better solution would be
                # better.
                A = initialtensors.build_complexion(
                    ham, pars, complexion_step_direction=-1,
                    complexion_timestep=step,
                    complexion_spacestep=pars["euclideon_spacestep"],
                    complexion_padding=pars["euclideon_padding"],
                    complexion_eps=pars["euclideon_eps"],
                    complexion_chis=pars["euclideon_chis"]
                )
                msg = "\nDecreasing step to {}, A.shape: {}".format(
                    step, A.shape
                )
                logging.info(msg)

    umps.canonicalize(crude=False, confirm=True)
    return umps

