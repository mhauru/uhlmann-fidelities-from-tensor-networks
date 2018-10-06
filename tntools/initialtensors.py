import numpy as np
import itertools as itt
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
import logging
from ncon import ncon
from tensors import Tensor
from tensors import TensorZ2, TensorZ3, TensorU1

""" Module for getting the initial tensors for different models.
Uses the tensors package.

The user is expected to call the function get_initial_tensor with
a dictionary as an argument that holds the necessary parameters,
including "model" and things like "beta" or various couplings.
The values of model that are supported, at least to some degree, are
ising: Classical square-lattice Ising model
potts3: Classical square-lattice 3-state Potts model
sixvertex The six-vertex model
ising3d: Classical cubical lattice Ising model
potts33d: Classical cubical lattice 3-state Potts model
Also included in the dictionary should be a boolean for
"symmetry_tensors", which determines whether symmetry preserving tensors
are to be used or not.

Some other functions are included, for instance for getting impurity
tensors for topological defects of the square lattice Ising model.
See the source code.
"""

# TODO: The 2D part is acceptable, but the 3D stuff is a big ad hoc
# mess. Also, everything needs to be documented.

# # # # # # # # # # # # # 2D models # # # # # # # # # # # # # # # # #

def ising_hamiltonian(pars):
    ham = (- pars["J"]*np.array([[ 1,-1],
                                 [-1, 1]],
                                dtype=pars["dtype"])
           + pars["H"]*np.array([[-1, 0],
                                 [ 0, 1]],
                                dtype=pars["dtype"]))
    return ham

def potts3_hamiltonian(pars):
    ham = -pars["J"]*np.eye(3, dtype=pars["dtype"])
    return ham

hamiltonians = {}
hamiltonians["ising"] = ising_hamiltonian
hamiltonians["potts3"] = potts3_hamiltonian

symmetry_classes_dims_qims = {}
symmetry_classes_dims_qims["ising"] = (TensorZ2, [1,1], [0,1])
symmetry_classes_dims_qims["potts3"] = (TensorZ3, [1,1,1], [0,1,2])

# Transformation matrices to the bases where the symmetry is explicit.
symmetry_bases = {}
symmetry_bases["ising"] = np.array([[1, 1],
                                    [1,-1]]) / np.sqrt(2)
phase = np.exp(2j*np.pi/3)
symmetry_bases["potts3"] = np.array([[1,       1,         1],
                                     [1,    phase, phase**2],
                                     [1, phase**2,    phase]],
                                    dtype=np.complex_) / np.sqrt(3)
del(phase)


def get_initial_tensor(pars, **kwargs):
    if kwargs:
        pars = pars.copy()
        pars.update(kwargs)
    model_name = pars["model"].strip().lower()
    if model_name == "sixvertex":
        return get_initial_sixvertex_tensor(pars)
    elif model_name == "ising3d":
        return get_initial_tensor_ising_3d(pars)
    elif model_name == "potts33d":
        return get_initial_tensor_potts33d(pars)
    elif model_name == "complexion_qising":
        ham = get_ham(pars, model="qising")
        complexion = build_complexion(ham, pars)
        return complexion
    elif model_name == "complexion_qising_tricrit":
        ham = get_ham(pars, model="qising_tricrit")
        complexion = build_complexion(ham, pars)
        return complexion
    elif model_name == "complexion_sq_qising":
        ham = get_ham(pars, model="qising")
        complexion = build_complexion(ham, pars, square_hamiltonian=True)
        return complexion
    else:
        ham = hamiltonians[model_name](pars)
        boltz = np.exp(-pars["beta"]*ham)
        A_0 = np.einsum('ab,bc,cd,da->abcd', boltz, boltz, boltz, boltz)
        u = symmetry_bases[model_name]
        u_dg = u.T.conjugate()
        A_0 = ncon((A_0, u, u, u_dg, u_dg),
                   ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
        if pars["symmetry_tensors"]:
            cls, dim, qim = symmetry_classes_dims_qims[model_name]
            A_0 = cls.from_ndarray(A_0, shape=[dim]*4, qhape=[qim]*4,
                                   dirs=[1,1,-1,-1])
        else:
            A_0 = Tensor.from_ndarray(A_0)
    return A_0


def get_initial_sixvertex_tensor(pars):
    try:
        a = pars["sixvertex_a"]
        b = pars["sixvertex_b"]
        c = pars["sixvertex_c"]
    except KeyError:
        u = pars["sixvertex_u"]
        lmbd = pars["sixvertex_lambda"]
        rho = pars["sixvertex_rho"]
        a = rho*np.sin(lmbd - u)
        b = rho*np.sin(u)
        c = rho*np.sin(lmbd)
    A_0 = np.zeros((2,2,2,2), dtype=pars["dtype"])
    A_0[1,0,0,1] = a
    A_0[0,1,1,0] = a
    A_0[0,0,1,1] = b
    A_0[1,1,0,0] = b
    A_0[0,1,0,1] = c
    A_0[1,0,1,0] = c
    if pars["symmetry_tensors"]:
        dim = [1,1]
        qim = [-1,1]
        A_0 = TensorU1.from_ndarray(A_0, shape=[dim]*4, qhape=[qim]*4,
                                    dirs=[1,1,1,1])
        A_0 = A_0.flip_dir(2)
        A_0 = A_0.flip_dir(3)
    else:
        A_0 = Tensor.from_ndarray(A_0)
    return A_0


def get_KW_tensor(pars):
    """ The Kramers-Wannier duality defect of the classical 2D
    square lattice Ising model.
    """
    eye = np.eye(2, dtype=np.complex_)
    ham = hamiltonians["ising"](pars)
    B = np.exp(-pars["beta"] * ham)
    H = np.array([[1,1], [1,-1]], dtype=np.complex_)/np.sqrt(2)
    y_trigged = np.ndarray((2,2,2), dtype=np.complex_)
    y_trigged[:,:,0] = eye
    y_trigged[:,:,1] = sigma('y')
    D_sigma = np.sqrt(2) * np.einsum('ab,abi,ic,ad,adk,kc->abcd',
                                     B, y_trigged, H,
                                     B, y_trigged.conjugate(), H)

    u = symmetry_bases["ising"]
    u_dg = u.T.conjugate()
    D_sigma = ncon((D_sigma, u, u, u_dg, u_dg),
                   ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
    if pars["symmetry_tensors"]:
        D_sigma = TensorZ2.from_ndarray(D_sigma, shape=[[1,1]]*4,
                                        qhape=[[0,1]]*4, dirs=[1,1,-1,-1])
    else:
        D_sigma = Tensor.from_ndarray(D_sigma, dirs=[1,1,-1,-1])
    return D_sigma


def get_KW_unitary(pars):
    """ The unitary that moves the Kramers-Wannier duality defect of the
    classical 2D square lattice Ising model.
    """
    eye = np.eye(2, dtype=np.complex_)
    CZ = Csigma_np("z")
    U = ncon((CZ,
              R(np.pi/4, 'z'), R(np.pi/4, 'x'),
              R(np.pi/4, 'y')),
             ([-1,-2,5,6],
              [-3,5], [3,6],
              [-4,3]))
    u = symmetry_bases["ising"]
    u_dg = u.T.conjugate()
    U = ncon((U, u, u_dg, u_dg, u),
             ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
    U *= -1j
    if pars["symmetry_tensors"]:
        U = TensorZ2.from_ndarray(U, shape=[[1,1]]*4, qhape=[[0,1]]*4,
                                  dirs=[1,1,-1,-1])
    else:
        U = Tensor.from_ndarray(U, dirs=[1,1,1,-1,-1,-1])
    return U


def Csigma_np(sigma_str):
    eye = np.eye(2, dtype=np.complex_)
    CNOT = np.zeros((2,2,2,2), dtype=np.complex_)
    CNOT[:,0,:,0] = eye
    CNOT[:,1,:,1] = sigma(sigma_str)
    return CNOT


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


def R(alpha, c):
    s = sigma(c)
    eye = np.eye(2, dtype=np.complex_)
    res = np.cos(alpha)*eye + 1j*np.sin(alpha)*s
    return res


# # # # # # # # Quantum complexions and hamiltonians # # # # # # # # # #


def get_ham(pars, **kwargs):
    if kwargs:
        pars = pars.copy()
        pars.update(kwargs)
    model_name = pars["model"].strip().lower()
    if model_name in {"qising", "complexion_qising"}:
        ham = qising_ham(pars)
    elif model_name in {"qising_tricrit", "complexion_qising_tricrit"}:
        ham_ising = qising_ham(pars)
        ham_pert = tricrit_perturbation_ham(pars)
        eye = type(ham_ising).eye(2, qim=[0,1])
        ham = (ncon((ham_ising, eye), ([-1,-2,-11,-12], [-3,-13]))
               + ncon((eye, ham_ising), ([-1,-11], [-2,-3,-12,-13]))
               + pars["tricrit_perturbation_factor"]*ham_pert)
    else:
        msg = ("Don't know how to generate ham for {}.".format(model_name))
        raise ValueError(msg)
    return ham


def qising_ham(pars):
    h_trans = pars["h_trans"]
    h_long = pars["h_long"]
    eye2 = np.eye(2)
    ham = (- ncon((sigma('x'), sigma('x')), ([-1,-11], [-2,-12]))
           - h_trans/2*ncon((eye2, sigma('z')), ([-1,-11], [-2,-12]))
           - h_trans/2*ncon((sigma('z'), eye2), ([-1,-11], [-2,-12]))
           - h_long/2*ncon((sigma('x'), eye2), ([-1,-11], [-2,-12]))
           - h_long/2*ncon((eye2, sigma('x')), ([-1,-11], [-2,-12]))
           + 4/np.pi*ncon((eye2, eye2), ([-1,-11], [-2,-12]))
           )/2
    dim, qim = [1,1], [0,1]
    # TODO What's the purpose of the check of the model?
    if pars["symmetry_tensors"] and pars["model"] == "qising":
        tensor_cls = TensorZ2
    else:
        tensor_cls = Tensor
    ham = tensor_cls.from_ndarray(ham, shape=[dim]*4, qhape=[qim]*4,
                                  dirs=[1,1,-1,-1])
    return ham


def tricrit_perturbation_ham(pars):
    ham = (ncon((sigma('z'), sigma('x'), sigma('x')),
                ([-1,-11], [-2,-12], [-3,-13]))
           + ncon((sigma('x'), sigma('x'), sigma('z')),
                  ([-1,-11], [-2,-12], [-3,-13])))
    dim, qim = [1,1], [0,1]
    # TODO What's the purpose of the check of the model?
    if pars["symmetry_tensors"] and pars["model"] == "qising":
        tensor_cls = TensorZ2
    else:
        tensor_cls = Tensor
    ham = tensor_cls.from_ndarray(ham, shape=[dim]*6, qhape=[qim]*6,
                                  dirs=[1,1,1,-1,-1,-1])
    return ham


def build_qham_open(ham, N):
    T = type(ham)
    dim = ham.shape[0]
    qim = ham.qhape[0] if ham.qhape is not None else None
    ham = ham.to_ndarray()
    k = len(ham.shape)//2
    d = ham.shape[0]
    ham = np.reshape(ham, (d**k, d**k))
    eye = np.eye(d)
    ids = 1.
    result = ham
    for i in range(k+1, N+1):
        ids = np.kron(ids, eye)
        result = np.kron(result, eye)
        result += np.kron(ids, ham)
    result = np.reshape(result, (d,)*(2*N))
    result = T.from_ndarray(result, shape=[dim]*(2*N), qhape=[qim]*(2*N),
                            dirs=([1]*N + [-1]*N))
    return result


# TODO Should this really be in initialtensors.py?
def exp_op(A):
    T = type(A)
    shape = A.shape
    qhape = A.qhape
    dirs = A.dirs
    A = A.to_ndarray()
    N = int(len(A.shape)/2)
    d = A.shape[0]
    A = np.reshape(A, (d**N, d**N))
    EA = spla.expm(A)
    EA = np.reshape(EA, (d,)*(2*N))
    EA = T.from_ndarray(EA, shape=shape, qhape=qhape, dirs=dirs)
    return EA


def build_complexion(ham, pars, square_hamiltonian=False, **kwargs):
    if kwargs:
        pars = pars.copy()
        pars.update(kwargs)
    timestep = pars["complexion_timestep"]
    spacestep = pars["complexion_spacestep"]
    padding = pars["complexion_padding"]
    spacestep = int(np.ceil(spacestep))
    halfN = spacestep + padding
    N = halfN*2
    M = N + spacestep
    unit = pars["complexion_step_direction"]

    HN = build_qham_open(ham, N)
    if square_hamiltonian:
        inds1 = [-i for i in range(1,N+1)] + [i for i in range(1,N+1)]
        inds2 = [i for i in range(1,N+1)] + [-i for i in range(N+1,2*N+1)]
        HN = ncon((HN, HN), (inds1, inds2))

    UN = exp_op(unit*timestep*HN)

    U, S, V, error = UN.svd(
        list(range(0,halfN)) + list(range(N,3*halfN)),
        list(range(halfN,N)) + list(range(3*halfN,2*N)),
        eps=pars["complexion_eps"], chis=pars["complexion_chis"],
        return_rel_err=True
    )

    HM = build_qham_open(ham, M)
    if square_hamiltonian:
        inds1 = [-i for i in range(1,M+1)] + [i for i in range(1,M+1)]
        inds2 = [i for i in range(1,M+1)] + [-i for i in range(M+1,2*M+1)]
        HM = ncon((HM, HM), (inds1, inds2))
    UM = exp_op(unit*timestep*HM)

    Uindices = list(range(1,N+1)) + [-1]
    UMindices = (
        list(range(1,halfN+1))
        + [-i for i in range(2,2+spacestep)] + list(range(N+1,3*halfN+1))
        + list(range(halfN+1,N+1))
        + [-i for i in range(3+spacestep,3+spacestep*2)]
        + list(range(3*halfN+1,2*N+1))
    )
    Vindices = [-2-spacestep] + list(range(N+1,2*N+1))
    complexion = ncon((U.conjugate(), UM, V.conjugate()),
                      (Uindices, UMindices, Vindices))
    complexion = complexion.join_indices(
        list(range(1,1+spacestep)),
        list(range(2+spacestep,2+2*spacestep)),
        dirs=[1,-1]
    )
    try:
        S_isqrt = S**(-1/2)
    except ZeroDivisionError:
        S_isqrt = S.copy()
        for k, v in S_isqrt.sects.items():
            S_isqrt[k] = v**(-1/2)
    complexion = complexion.multiply_diag(S_isqrt, 0, direction="left")
    complexion = complexion.multiply_diag(S_isqrt, 2, direction="right")
    shp = type(complexion).flatten_shape(complexion.shape)
    if pars["verbosity"] > 0:
        logging.info("Built complexion with shape {}, error {}"
                     .format(shp, error))
    return complexion



# # # # # # # # # # # # # 3D models # # # # # # # # # # # # # # # # #
# TODO: Incorporate this into the more general framework.
# TODO: Implement this for symmetry preserving tensors.

def get_initial_tensor_CDL_3d(pars):
    delta = np.eye(2, dtype = pars["dtype"])
    A = np.einsum(('ae,fi,jm,nb,cq,rk,lu,vd,gs,to,pw,xh '
                   '-> abcdefghijklmnopqrstuvwx'),
                  delta, delta, delta, delta, delta, delta,
                  delta, delta, delta, delta, delta, delta)
    return Tensor.from_ndarray(A.reshape((16,16,16,16,16,16)))


def get_initial_tensor_CDL_3d_v2(pars):
    delta = np.eye(2, dtype = pars["dtype"])
    A = ncon((delta,)*12,
             ([-11,-21], [-12,-41], [-13,-51], [-14,-61],
              [-31,-22], [-32,-42], [-33,-52], [-34,-62],
              [-23,-63], [-64,-43], [-44,-53], [-54,-24]))
    return Tensor.from_ndarray(A.reshape((16,16,16,16,16,16)))


def get_initial_tensor_CQL_3d(pars):
    delta = np.array([[[1,0],[0,0]],[[0,0],[0,1]]])
    A = np.einsum(('aeu,fiv,gjq,hbr,mcw,nxk,ols,ptd '
                   '-> abcdefghijklmnopqrstuvwx'),
                  delta, delta, delta, delta, delta, delta, delta,
                  delta)
    return Tensor.from_ndarray(A.reshape((16,16,16,16,16,16)))


def get_initial_tensor_ising_3d(pars):
    beta = pars["beta"]
    ham = ising3d_ham(beta)
    A_0 = np.einsum('ai,aj,ak,al,am,an -> ijklmn',
                    ham, ham, ham, ham, ham, ham)
    if pars["symmetry_tensors"]:
        cls, dim, qim = TensorZ2, [1,1], [0,1]
        A_0 = cls.from_ndarray(A_0, shape=[dim]*6, qhape=[qim]*6,
                               dirs=[1,1,-1,-1,1,-1])
    else:
        A_0 = Tensor.from_ndarray(A_0)
    return A_0


def get_initial_tensor_potts33d(pars):
    beta = pars["beta"]
    Q = potts_Q(beta, 3)
    A = np.einsum('ai,aj,ak,al,am,an -> ijklmn',
                  Q, Q, Q.conjugate(), Q.conjugate(), Q, Q.conjugate())
    if np.linalg.norm(np.imag(A)) < 1e-12:
        A = np.real(A)
    if pars["symmetry_tensors"]:
        cls, dim, qim = symmetry_classes_dims_qims["potts3"]
        A = cls.from_ndarray(A, shape=[dim]*6, qhape=[qim]*6,
                             dirs=[1,1,-1,-1,1,-1])
    else:
        A = Tensor.from_ndarray(A)
    return A


def potts_Q(beta, q):
    Q = np.zeros((q,q), np.complex_)
    for i, j in itt.product(range(q), repeat=2):
        Q[i,j] = (np.exp(1j*2*np.pi*i*j/q)
                  * np.sqrt((np.exp(beta) - 1 + (q if j==0 else 0))/ q))
    return Q


def potts_Q_inv(beta, q):
    q = 3
    Q = np.zeros((q,q), np.complex_)
    for i, j in itt.product(range(q), repeat=2):
        Q[i,j] = (np.exp(-1j*2*np.pi*i*j/q)
                  * np.sqrt(1/(q*(np.exp(beta) - 1 + (q if i==0 else 0)))))
    return Q


# # # 3D impurities # # #


impurity_dict = dict()

# 3D Ising
ising_dict = {
    "id": np.eye(2),
    "sigmax": np.real(sigma("x")),
    "sigmay": sigma("y"),
    "sigmaz": np.real(sigma("z"))
}
for k, M in ising_dict.items():
    u = symmetry_bases["ising"]
    u_dg = u.T.conjugate()
    M = ncon((M, u, u_dg),
             ([1,2], [-1,1], [-2,2]))
    cls, dim, qim = symmetry_classes_dims_qims["ising"]
    M = cls.from_ndarray(M, shape=[dim]*2, qhape=[qim]*2,
                         dirs=[-1,1])
    ising_dict[k] = lambda pars: M
impurity_dict["ising"] = ising_dict
del(ising_dict)

impurity_dict["ising3d"] = dict()
impurity_dict["ising3d"]["id"] = lambda pars: TensorZ2.eye([1,1]).transpose()
impurity_dict["ising3d"]["sigmaz"] = lambda pars: (
    TensorZ2.from_ndarray(sigmaz("z"), shape=[[1,1]]*2, qhape=[[0,1]]*2,
                          dirs=[-1,1])
)
impurity_dict["ising3d"]["sigmax"] = lambda pars: (
    TensorZ2.from_ndarray(sigmaz("x"), shape=[[1,1]]*2, qhape=[[0,1]]*2,
                          dirs=[-1,1])
)
impurity_dict["ising3d"]["sigmay"] = lambda pars: (
    TensorZ2.from_ndarray(sigmaz("y"), shape=[[1,1]]*2, qhape=[[0,1]]*2,
                          dirs=[-1,1])
)

def ising3d_ham(beta):
    res = np.array([[np.cosh(beta)**0.5,  np.sinh(beta)**0.5],
                    [np.cosh(beta)**0.5, -np.sinh(beta)**0.5]])
    return res

def ising3d_ham_inv(beta):
    res = 0.5*np.array([[np.cosh(beta)**(-0.5),  np.cosh(beta)**(-0.5)],
                        [np.sinh(beta)**(-0.5), -np.sinh(beta)**(-0.5)]])
    return res

def ising3d_ham_T(beta):
    res = np.array([[np.cosh(beta)**0.5,  np.cosh(beta)**0.5],
                    [np.sinh(beta)**0.5, -np.sinh(beta)**0.5]])
    return res

def ising3d_ham_T_inv(beta):
    res = 0.5*np.array([[np.cosh(beta)**(-0.5),  np.sinh(beta)**(-0.5)],
                        [np.cosh(beta)**(-0.5), -np.sinh(beta)**(-0.5)]])
    return res

def ising3d_U(beta):
    matrix = (ising3d_ham_inv(beta)
              .dot(sigma("z"))
              .dot(ising3d_ham(beta))
              .dot(ising3d_ham_T(beta))
              .dot(sigma("z"))
              .dot(ising3d_ham_T_inv(beta)))
    matrix = np.real(matrix)
    matrix = TensorZ2.from_ndarray(matrix, shape=[[1,1]]*2, qhape=[[0,1]]*2,
                                   dirs=[-1,1])
    # Factor of -1 because U = - \partial log Z / \partial beta, and a
    # factor of 3 because there are two bonds per lattice site, and we
    # normalize by number of sites.
    matrix *= -3
    return matrix

impurity_dict["ising3d"]["U"] = lambda pars: ising3d_U(pars["beta"])


# 3D Potts3
impurity_dict["potts33d"] = dict()

def potts33d_U(beta):
    Q = potts_Q(beta, 3)
    energymat = (Q.dot(Q.conjugate().transpose()) * np.eye(Q.shape[0]))
    matrix = (potts_Q_inv(beta, 3)
              .dot(energymat)
              .dot(potts_Q_inv(beta, 3).conjugate().transpose()))
    if np.linalg.norm(np.imag(matrix)) < 1e-12:
        matrix = np.real(matrix)
    cls, dim, qim = symmetry_classes_dims_qims["potts3"]
    matrix = cls.from_ndarray(matrix, shape=[dim]*2,
                              qhape=[qim]*2, dirs=[-1,1])
    return matrix

impurity_dict["potts33d"]["U"] = lambda pars: potts33d_U(pars["beta"])


def get_initial_impurity(pars, legs=(3,), factor=3, **kwargs):
    if kwargs:
        pars = pars.copy()
        pars.update(kwargs)
    A_pure = get_initial_tensor(pars)
    model = pars["model"]
    impurity = pars["impurity"]
    try:
        impurity_matrix = impurity_dict[model][impurity](pars)
    except KeyError:
        msg = ("Unknown (model, impurity) combination: ({}, {})"
               .format(model, impurity))
        raise ValueError(msg)
    # TODO The expectation that everything is in the symmetry basis
    # clashes with how 2D ising and potts initial tensors are generated.
    if not pars["symmetry_tensors"]:
        impurity_matrix = Tensor.from_ndarray(impurity_matrix.to_ndarray())
    impurity_matrix *= -1
    A_impure = 0
    if 0 in legs:
        A_impure += ncon((A_pure, impurity_matrix),
                         ([1,-2,-3,-4,-5,-6], [1,-1]))
    if 1 in legs:
        A_impure += ncon((A_pure, impurity_matrix),
                         ([-1,2,-3,-4,-5,-6], [2,-2]))
    if 2 in legs:
        A_impure += ncon((A_pure, impurity_matrix.transpose()),
                         ([-1,-2,3,-4,-5,-6], [3,-3]))
    if 3 in legs:
        A_impure += ncon((A_pure, impurity_matrix.transpose()),
                         ([-1,-2,-3,4,-5,-6], [4,-4]))
    if 4 in legs:
        A_impure += ncon((A_pure, impurity_matrix),
                         ([-1,-2,-3,-4,5,-6], [5,-5]))
    if 5 in legs:
        A_impure += ncon((A_pure, impurity_matrix.transpose()),
                         ([-1,-2,-3,-4,-5,6], [6,-6]))
    A_impure *= factor
    return A_impure


