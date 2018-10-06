import numpy as np
import itertools as it
import scipy.integrate as integrate

""" Module that gets exact data such as free energies and scaling
dimensions for known models. Most functions expect as arguments a
dictionary called pars, where one of the keys would be called "model"
and the value for that would be the name of the model. Other necessary
things may be expected to be included in the same dictionary, such
as a (key, value) pair for the inverse temperature.

Supported at the moment are only 2D classical square-lattice models, namely
the Ising, 3-state Potts, and six-vertex models. The model names used
for these are "ising", "potts3" and "sixvertex".
"""


##################### Critical points #####################

def get_critical_beta(pars, **kwargs):
    pars = update_pars(pars, **kwargs)
    model_name = pars["model"].strip().lower()
    if model_name == "ising":
        if pars["H"] != 0:
            raise NotImplementedError("Critical beta of Ising model with "
                                      "H != 0 not implemented.")
        beta_c = np.log(1 + np.sqrt(2)) / (2*pars["J"])
    elif model_name == "potts3":
        beta_c = np.log(np.sqrt(3) + 1) / pars["J"]
    else:
        beta_c = None
    return beta_c

##################### CFT data #####################

def get_central_charge(pars, **kwargs):
    # TODO Should this check that we are at criticality?
    pars = update_pars(pars, **kwargs)
    model_name = pars["model"].strip().lower()
    if model_name == "ising":
        c = 1/2
    elif model_name == "potts3":
        c = 4/5
    elif model_name == "sixvertex":
        c = 1
    return c


# The different possible (anti)holomorphic parts of primary operators.
# The first element is the dimension, the next one is a list of Virasoro
# characters.
ising_0 = (0, (1,0,1,1,2,2,3,3,5,5,6,7,11,12,16,18))
ising_116 = (1/16, (1,1,1,2,2,3,4,5,6,8,10,12,15,18,22,27))
ising_12 = (1/2, (1,1,1,1,2,2,3,4,5,6,8,9,12,14,17,20))

potts3_0 = (0, (1,0,1,1,2,2,4,4,7,8,12,14,21,24,34,41))
potts3_18 = (1/8, (1,1,1,2,3,4,6,8,11,15,20,26,35,45,58,75))
potts3_23 = (2/3, (1,1,2,2,4,5,8,10,15,19,27,34,46,58,77,96))
potts3_138 = (13/8, (1,1,2,3,4,6,9,12,16,22,29,38,50,64,82,105))
potts3_25 = (2/5, (1,1,1,2,3,4,6,8,11,15,20,26,35,45,58,74))
potts3_140 = (1/40, (1,1,2,3,4,6,9,12,17,23,31,41,54,70,91,117))
potts3_115 = (1/15, (1,1,2,3,5,7,10,14,20,26,36,47,63,81,106,135))
potts3_2140 = (21/40, (1,1,2,3,5,7,10,14,20,26,36,47,63,81,106,135))
potts3_75 = (7/5, (1,1,2,2,4,5,8,10,15,19,26,33,45,56,74,92))
potts3_3 = (3, (1,1,2,3,4,5,8,10,14,18,24,31,41,51,66,83))


def get_primary_data(maxi, pars, alpha, qnum=None, **kwargs):
    pars = update_pars(pars, **kwargs)
    modelname = pars["model"].strip().lower()
    prima_pairs = ()
    if modelname == "ising" and "KW" in pars and pars["KW"]:
        if qnum is None or qnum == 0:
            prima_pairs += ((ising_0, ising_116),
                            (ising_12, ising_116))
        if qnum is None or qnum == 1:
            prima_pairs += ((ising_116, ising_0),
                            (ising_116, ising_12))
    elif modelname == "ising" and "g" in pars and np.allclose(pars["g"], 0):
        # Because there really isn't an antiholomorphic part, we use a
        # trivial (0, (1,0,0,...)) to create the right dims and degs.
        if qnum is None or qnum == 0:
            prima_pairs += ((ising_0, (0, (1,) + (0,)*len(ising_0))),)
        if qnum is None or qnum == 1:
            prima_pairs += ((ising_12, (0, (1,) + (0,)*len(ising_12))),)
    elif modelname == "ising" and not alpha:
        if qnum is None or qnum == 0:
            prima_pairs += ((ising_0, ising_0),
                           (ising_12, ising_12))
        if qnum is None or qnum == 1:
            prima_pairs += ((ising_116, ising_116),)
    elif modelname == "ising" and alpha and np.allclose(np.abs(alpha), np.pi):
        if qnum is None or qnum == 0:
            prima_pairs += ((ising_116, ising_116),)
        if qnum is None or qnum == 1:
            prima_pairs += ((ising_12, ising_0),
                           (ising_0, ising_12))
    elif modelname == "potts3" and not alpha:
        if qnum is None or qnum == 0:
            prima_pairs += ((potts3_0, potts3_0),
                           (potts3_25, potts3_25),
                           (potts3_75, potts3_75),
                           (potts3_3, potts3_3),
                           (potts3_25, potts3_75),
                           (potts3_75, potts3_25),
                           (potts3_3, potts3_0),
                           (potts3_0, potts3_3))
        if qnum is None or qnum == 1 or qnum == 2:
            prima_pairs += ((potts3_115, potts3_115),
                            (potts3_23, potts3_23))
    elif modelname == "potts3" and alpha and np.allclose(alpha, np.pi*2/3):
        if qnum is None or qnum == 0:
            prima_pairs += ((potts3_115, potts3_115),
                            (potts3_23, potts3_23))
        if qnum is None or qnum == 1:
            prima_pairs += ((potts3_25, potts3_115),
                            (potts3_0, potts3_23),
                            (potts3_75, potts3_115),
                            (potts3_3, potts3_23))
        if qnum is None or qnum == 2:
            prima_pairs += ((potts3_115, potts3_25),
                            (potts3_23, potts3_0),
                            (potts3_115, potts3_75),
                            (potts3_23, potts3_3))
    elif modelname == "potts3" and alpha and np.allclose(alpha, np.pi*4/3):
        if qnum is None or qnum == 0:
            prima_pairs += ((potts3_115, potts3_115),
                            (potts3_23, potts3_23))
        if qnum is None or qnum == 1:
            prima_pairs += ((potts3_115, potts3_25),
                            (potts3_23, potts3_0),
                            (potts3_115, potts3_75),
                            (potts3_23, potts3_3))
        if qnum is None or qnum == 2:
            prima_pairs += ((potts3_25, potts3_115),
                            (potts3_0, potts3_23),
                            (potts3_75, potts3_115),
                            (potts3_3, potts3_23))
    scaldims, spins, degs = build_primas(prima_pairs)
    # Truncate and sort.
    truncated_triples = it.takewhile(lambda t: t[0] < maxi,
                                     sorted(zip(scaldims, spins, degs)))
    scaldims, spins, degs = tuple(zip(*truncated_triples))
    # Group together identical (dim, spin) pairs and sum up the
    # degeneracies.
    grouped_dims, grouped_spins, grouped_degs = [], [], []
    for dim, spin, deg in zip(scaldims, spins, degs):
        if (grouped_dims
                and grouped_dims[-1] == dim
                and grouped_spins[-1] == spin):
            grouped_degs[-1] += 1
        else:
            grouped_dims.append(dim)
            grouped_spins.append(spin)
            grouped_degs.append(deg)
    return grouped_dims, grouped_spins, grouped_degs


def build_primas(prima_pairs):
    """ Given a list of pairs of holomorphic and anti-holomorphic parts
    of primaries, construct the scaling dimensions, spins and
    degeneracies of the towers of these primaries.
    """
    scaldims = []
    spins = []
    degs = []
    for left, right in prima_pairs:
        # Virs refer to Virasoro characters.
        h, Virs_h = left
        hbar, Virs_hbar = right
        i_Virs_h = enumerate(Virs_h)
        i_Virs_hbar = enumerate(Virs_hbar)
        for (i, Vir_h), (j, Vir_hbar) in it.product(i_Virs_h, i_Virs_hbar):
            deg = Vir_h*Vir_hbar
            if deg > 0:
                scaldims.append(h+i + hbar+j)
                spins.append(h+i - hbar-j)
                degs.append(deg)
    # We make sure that values that are the same up to numerical errors
    # are stored as exactly the same. This is necessary later, when
    # sorting is done on these values.
    for li in (scaldims, spins, degs):
        past = set()
        for i, el in enumerate(li):
            for old_el in past:
                if abs(el-old_el) < 1e-7:
                    # The elements are considered the same and made
                    # exactly equal.
                    el = old_el
                    break
            past.add(el)
            li[i] = el
    return scaldims, spins, degs


def get_scaling_dimensions(m, pars, alpha=0, **kwargs):
    scaldims, spins, degs = get_primary_data(m, pars, alpha=alpha, **kwargs)
    return scaldims


def get_conformal_spins(m, pars, alpha=0, **kwargs):
    scaldims, spins, degs = get_primary_data(m, pars, alpha=alpha, **kwargs)
    return spins

##################### Free energy #####################

def get_free_energy(pars, **kwargs):
    pars = update_pars(pars, **kwargs)
    model_name = pars["model"].strip().lower()
    if model_name == "ising":
        f = ising_exact_f(beta=pars["beta"], J=pars["J"], H=pars["H"])
    elif model_name == "potts3":
        f = potts3_exact_f(beta=pars["beta"], J=pars["J"])
    elif model_name == "sixvertex":
        f = sixvertex_exact_f(a=pars["sixvertex_a"], b=pars["sixvertex_b"],
                              c=pars["sixvertex_c"])
    return f


def ising_exact_f(beta, J, H):
    if H != 0:
        raise NotImplementedError("Free energy of Ising model with H != 0 "
                                  "not implemented.")
    sinh = np.sinh(2*beta*J)
    cosh = np.cosh(2*beta*J)
    def integrand(theta):
        res = np.log(cosh**2 +
                     sinh**2 * np.sqrt(1+sinh**(-4) -
                                       2*sinh**(-2)*np.cos(theta)))
        return res
    integral, err = integrate.quad(integrand, 0, np.pi)
    f = -(np.log(2)/2 + integral/(2*np.pi)) / beta
    return f

def potts3_exact_f(beta, J):
    if np.allclose(beta, get_critical_beta({}, J=J, model="potts3")):
        q = 3
        mu = np.arccos(np.sqrt(q)/2)
        def integrand(x):
            res = np.tanh(mu*x)*np.sinh((np.pi - mu)*x) / (x*np.sinh(np.pi*x))
            return res
        f = (-0.5*np.log(q)
             -integrate.quad(integrand, -100, 100, points=[0])[0])
        f = f / beta
    else:
        raise NotImplementedError()
    return f

def sixvertex_exact_f(a, b, c):
    raise NotImplementedError()
    # TODO This may not be correct at all, and the integration does not work.
    m = 0.5*(a+b+c)
    if a<m and b<m and c<m:
        eps_1 = -np.log(a)
        mu = np.arcsin(c)/2
        omega = mu - np.arcsin(a)
        def integrand(x):
            res = ((np.sinh(2*(mu+omega)*x) / np.sinh(np.pi*x)) *
                   (np.sinh((np.pi-2*mu)*x) / np.cosh(2*mu*x)) /
                   (2*x))
            return res
        f = eps_1 - integrate.quad(integrand, -100, 100, points=[0])[0]
    else:
        raise NotImplementedError()
    return f

##################### Utility funtions #####################

def update_pars(pars, **kwargs):
    pars = pars.copy()
    pars.update(kwargs)
    return pars

