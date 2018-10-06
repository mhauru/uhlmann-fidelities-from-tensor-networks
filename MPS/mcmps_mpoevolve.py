import numpy as np
import pickle
import datetime
from matplotlib import pyplot as plt
from ncon import ncon
from tntools import datadispenser
from .McMPS import McMPS
from .UMPS import UMPS

def mpoevolve(mcmps, mpo, pars):
    mcmps.absorb_mpo(mpo)
    mcmps.canonicalize(chis=pars["chis"], eps=pars["eps"])
    return mcmps


def sigma(c):
    if c=="x":
        res = np.array([[ 0, 1],
                        [ 1, 0]], dtype=np.complex_)
    elif c=="y":
        res = np.array([[ 0j,-1j],
                        [ 1j, 0j]], dtype=np.complex_)
    elif c=="z":
        res = np.array([[ 1, 0],
                        [ 0,-1]], dtype=np.complex_)
    return res


def qising_ham(h_trans=1, h_long=0):
    eye2 = np.eye(2)
    ham = (- ncon((sigma('x'), sigma('x')), ([-1,-11], [-2,-12]))
           - h_trans/2*ncon((eye2, sigma('z')), ([-1,-11], [-2,-12]))
           - h_trans/2*ncon((sigma('z'), eye2), ([-1,-11], [-2,-12]))
           - h_long/2*ncon((sigma('x'), eye2), ([-1,-11], [-2,-12]))
           - h_long/2*ncon((eye2, sigma('x')), ([-1,-11], [-2,-12]))
           + 4/np.pi*ncon((eye2, eye2), ([-1,-11], [-2,-12]))
           )/2
    return ham


def generate_A(pars, step_factor=1.0, lorentzian=False, **kwargs):
    if lorentzian:
        step_dir = 1j
    else:
        step_dir = -1
    A = datadispenser.get_data(
        dbname, "A", pars=pars, complexion_step_direction=step_dir,
        iter_count=0,
        complexion_timestep=pars["complexion_timestep"]*step_factor,
        **kwargs
    )[0]
    return A

    
def entanglement_entropy(w):
    w_sq = w**2
    w_sq /= w_sq.sum()
    ent = -(w_sq*w_sq.log()).sum()# / np.log(len(w))
    return ent


if __name__ == "__main__":
    pars = {"model": "complexion_qising",
            "chis": range(1,50),
            "eps": 1e-5,
            "h_trans": 1.0,
            "h_long": 0.0,
            "complexion_timestep": 1,
            "complexion_spacestep": 1,
            "complexion_eps": 1e-5,
            "complexion_chis": range(1,7)
            }
    dbname = "../data/"

    step_factor = 50
    min_step_factor = 1/8
    max_counter = 1000
    max_sub_counter = 15
    ent_eps = 1e-3
    energy_eps = 1e-3
    N = 10

    A = generate_A(pars, step_factor=step_factor)
    np.set_printoptions(suppress=True, linewidth=120)
    ham = qising_ham(h_trans=pars["h_trans"], h_long=pars["h_long"])
    ham = type(A).from_ndarray(ham)

    umps = UMPS.random(2, min(pars["chis"]), dtype=np.complex_, random_weights=True)
    tensors = []
    weightss = []
    for i in range(N):
        tensor = umps.tensor
        tensor = type(tensor).random(tensor.shape) + 1j*type(tensor).random(tensor.shape)
        tensors.append(tensor.copy())
        if i != N-1:
            weights = umps.weights
            weights = type(weights).random(weights.shape, invar=False)
            weightss.append(weights.copy())
    mcmps = McMPS(umps, tensors, weightss)
    print(mcmps.expect_local(ham, int(N/2)))

    energy = np.inf
    ent = 0
    counter = 0
    sub_counter = 0

    while counter < max_counter:
        sub_counter += 1
        counter += 1
        N = mcmps.length()
        centre = int(N/2)
        print()
        print("Length:", N)
        print("Bond dimension:", mcmps.virtualdim(centre))
        print("Counter: {} ({})".format(counter, sub_counter))
        print("Time step:", step_factor)
        mpoevolve(mcmps, A, pars)
        old_energy = energy
        energy = np.real(mcmps.expect_local(ham, centre))
        energy_change = np.abs((energy - old_energy)/energy)
        print("Energy: {:.3e} ({:.3e})".format(energy, energy_change))
        w = mcmps.weights(centre+1/2)
        old_ent = ent
        ent = entanglement_entropy(w)
        ent_change = np.abs((ent - old_ent)/ent)
        print("Entropy: {:.6f} ({:.3e})".format(ent, ent_change))
        #print("Spectrum:")
        #print(w)
        if  ((ent_change < ent_eps and energy_change < energy_eps)
             or sub_counter > max_sub_counter):
            step_factor *= 0.8
            sub_counter = 0
            if step_factor < min_step_factor:
                break
            else:
                A = generate_A(pars, step_factor=step_factor)
                msg = "\nDecreasing step to {}, A.shape: {}".format(
                    step_factor, A.shape
                )
                print(msg)

    sigmax = np.array([[0,1], [1,0]])
    sigmax = mcmps.tensortype().from_ndarray(sigmax)

    dist = 1000
    twopoints = mcmps.expect_twopoint(sigmax, sigmax, dist)
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.semilogy(range(1, dist+1), np.abs(twopoints), marker="o")
    ax2.loglog(range(1, dist+1), np.abs(twopoints), marker="o")

    n = 1000
    exact_points = np.linspace(1, dist, n)
    exact_values = exact_points**(-1/8)
    exact_values *= np.abs(twopoints[0])/exact_values[0]
    ax1.semilogy(exact_points, np.abs(exact_values))
    ax2.loglog(exact_points, np.abs(exact_values))

    plt.show()

    input_write = input("Write to file?")
    if input_write.lower() == "y":
        datetime_str = datetime.datetime.strftime(datetime.datetime.now(),
                                                 '%Y-%m-%d_%H-%M-%S')
        filename = (
            "pickles/{}_ground_state_mcmps_{}_{}.p"
            .format(datetime_str, pars["h_trans"], max(pars["chis"]))
        )
        with open(filename, "wb") as f:
            pickle.dump(mcmps, f)
            print("Wrote to {}.".format(filename))

