import numpy as np
import heapq
import copy
import operator as opr
import itertools as itt
import functools as fct
import scipy.sparse.linalg as spsla
from ncon import ncon
from tensors import AbelianTensor
from tensors import Tensor


""" A module that allows doing eigenvalue and singular value
decompositions of tensor networks, using an interface similar to ncon,
without ever contractng the full network. Meant to be used with the 
tensors package that implements tensors with internal abelian
symmetries.

In other words, there's a common situation where one wants to implement
a function that maps a vector v to v contracted with some tensor
network, with the contraction order specified for speed, and then
perform "sparse" power-method-style decompositions on this function.
ncon_sparseeig provides a convenient one-function-call interfact for
doing the whole thing, for both regular and symmetric tensors.
"""

# TODO: This module could use cleaning up and documenting. If I recall
# correctly, there's some pretty ad hoc ugliness in here.

# Commonalities

def get_commons(tensor_list):
    errmsg = "tensor_list in ncon_sparseeig has inhomogenous "

    types = set(map(type, tensor_list))
    if len(types) > 1:
        raise ValueError(errmsg + "types.")
    commontype = types.pop()

    dtypes = set(t.dtype for t in tensor_list)
    commondtype = np.find_common_type(dtypes, [])

    qoduli = set(t.qodulus if hasattr(t, "qodulus") else None
                    for t in tensor_list)
    if len(qoduli) > 1:
        raise ValueError(errmsg + "qoduli.")
    commonqodulus = qoduli.pop()

    return commontype, commondtype, commonqodulus


def get_free_indexdata(tensor_list, index_list):
    """ Figure out the numbers, dims, qims and dirs of all the free
    indices of the network that we want the eigenvalues of.
    """
    inds = []
    dims = []
    qims = []
    dirs = []
    for t, l in zip(tensor_list, index_list):
        for j, k in enumerate(l):
            if k < 0:
                inds.append(k)
                dims.append(t.shape[j])
                if t.qhape is not None:
                    dirs.append(t.dirs[j])
                    qims.append(t.qhape[j])
                else:
                    dirs.append(None)
                    qims.append(None)
    inds, dims, qims, dirs = zip(*sorted(zip(inds, dims, qims, dirs),
                                         reverse=True))
    return inds, dims, qims, dirs


def get_oneside_indexdata(commontype, free_dims, free_qims, free_dirs,
                          side_inds):
    side_dims = list(map(free_dims.__getitem__, side_inds))
    side_qims = list(map(free_qims.__getitem__, side_inds))
    if side_qims[0] is None:
        side_qims = None
    side_dirs = list(map(free_dirs.__getitem__, side_inds))
    side_flatdims = list(map(commontype.flatten_dim, side_dims))
    return side_dims, side_qims, side_dirs, side_flatdims


def side_index_list(index_list, free_inds, right_inds):
    """ Flip the signs of the contraction indices for the vector. """
    c_inds = tuple(map(free_inds.__getitem__, right_inds))
    c_inds_set = set(c_inds)
    # Change the signs of the corresponding indices in index_list.
    index_list = [[-i if i in c_inds_set else i
                   for i in l]
                  for l in index_list]
    c_inds = list(map(opr.neg, c_inds))
    index_list.append(c_inds)
    return index_list


def get_qnums(right_qims, qodulus, qnums_do):
    all_qnums = map(sum, itt.product(*right_qims))
    if qodulus is not None:
        all_qnums = set(q % qodulus for q in all_qnums)
    else:
        all_qnums = set(all_qnums)
    if qnums_do:
        qnums = sorted(all_qnums & set(qnums_do))
    else:
        qnums = sorted(all_qnums)
    return qnums


def truncate_func(s, u=None, v=None, chis=None, eps=0, trunc_err_func=None,
                  norm_sq=None, return_error=False):
    chis = s.matrix_decomp_format_chis(chis, eps)
    if hasattr(s, "sects"):
        if trunc_err_func is None:
            trunc_err_func = fct.partial(type(s).default_trunc_err_func,
                                         norm_sq=norm_sq)
        # First, find what chi will be.
        s_flat = s.to_ndarray()
        s_flat = -np.sort(-np.abs(s_flat))
        chi, err = Tensor.find_trunc_dim(s_flat, chis=chis, eps=eps,
                                         trunc_err_func=trunc_err_func)

        # Find out which values to keep, i.e. how to distribute chi in
        # the different blocks.
        dim_sum = 0
        minusabs_next_els = []
        dims = {}
        for k, val in s.sects.items():
            heapq.heappush(minusabs_next_els, (-np.abs(val[0]), k))
            dims[k] = 0
        while(dim_sum < chi):
            try:
                minusabs_el_to_add, key = heapq.heappop(minusabs_next_els)
            except IndexError:
                # All the dimensions are fully included.
                break
            dims[key] += 1
            this_key_els = s[key]
            if dims[key] < len(this_key_els):
                next_el = this_key_els[dims[key]]
                heapq.heappush(minusabs_next_els, (-np.abs(next_el), key))
            dim_sum += 1

        # Truncate each block and create the dim for the new index.
        new_dim = []
        todelete = []
        for k in s.sects.keys():
            d = dims[k]
            new_dim.append(d)
            if d > 0:
                s[k] = s[k][:d]
            else:
                # Avoiding changing the dictionary during the loop.
                todelete.append(k)
        for k in todelete:
            del(s[k])

        if u is not None:
            todelete = []
            for k in u.sects.keys():
                klast = k[-1]
                d = dims[(klast,)]
                if d > 0:
                    u[k] = u[k][..., :d]
                else:
                    todelete.append(k)
            for k in todelete:
                del(u[k])

        if v is not None:
            todelete = []
            for k in v.sects.keys():
                k0 = k[0]
                d = dims[(k0,)]
                if d > 0:
                    v[k] = v[k][:d, ...]
                else:
                    todelete.append(k)
            for k in todelete:
                del(v[k])

        # Remove zero dimension sectors from qim.
        new_qim = s.qhape[0]
        new_dim = [d for d in new_dim if d > 0]
        for k, d in dims.items():
            if d == 0:
                new_qim = [q for q in new_qim if q != k[0]]

        s.shape = [new_dim]
        s.qhape = [new_qim]
        if u is not None:
            u.shape = u.shape[0:-1] + [new_dim]
            u.qhape = u.qhape[0:-1] + [new_qim]
        if v is not None:
            v.shape = [new_dim] + v.shape[1:]
            v.qhape = [new_qim] + v.qhape[1:]

    else:
        chi, err = type(s).find_trunc_dim(s, chis=chis, eps=eps,
                                          trunc_err_func=trunc_err_func)
        s = s[:chi]
        if u is not None:
            u = u[..., :chi]
        if v is not None:
            v = v[:chi, ...]
    retval = (s,)
    if u is not None:
        retval += (u,)
    if v is not None:
        retval += (v,)
    if return_error:
        retval += (err,)
    if len(retval) == 1:
        retval = retval[0]
    return retval


def common_preprocess(tensor_list, index_list, matvec_order, rmatvec_order,
                      matmat_order, left_inds, right_inds,
                      print_progress=False, chis=None, kwargs={}):
    tensor_list = list(tensor_list)
    index_list = list(index_list)
    left_inds = tuple(left_inds)
    right_inds = tuple(right_inds)

    commontype, commondtype, commonqodulus = get_commons(tensor_list)

    free_inds, free_dims, free_qims, free_dirs = get_free_indexdata(
        tensor_list, index_list
    )

    left_dims, left_qims, left_dirs, left_flatdims = get_oneside_indexdata(
        commontype, free_dims, free_qims, free_dirs, left_inds
    )
    left_flatdim = fct.reduce(opr.mul, left_flatdims)

    right_dims, right_qims, right_dirs, right_flatdims = get_oneside_indexdata(
        commontype, free_dims, free_qims, free_dirs, right_inds
    )
    right_flatdim = fct.reduce(opr.mul, right_flatdims)

    # Flip the signs of the contraction indices for the vector.
    matvec_index_list = side_index_list(index_list, free_inds, right_inds)
    rmatvec_index_list = side_index_list(index_list, free_inds, left_inds)
    matmat_index_list = copy.deepcopy(matvec_index_list)
    minindex = min(min(l) for l in matmat_index_list)
    matmat_index_list[-1].append(minindex-1)

    # The permutation on the final legs.
    left_perm = list(np.argsort(left_inds))
    right_perm = list(np.argsort(right_inds))

    tensor_list_conj = [t.conjugate() for t in tensor_list]

    try:
        neg_right_dirs = list(map(opr.neg, right_dirs))
    except TypeError:
        neg_right_dirs = None
    try:
        neg_left_dirs = list(map(opr.neg, left_dirs))
    except TypeError:
        neg_left_dirs = None

    # TODO could we initialize an initial guess of commontype and avoid
    # all the to/from ndarray?
    def matvec(v, charge=0):
        v = np.reshape(v, right_flatdims)
        v = commontype.from_ndarray(v, shape=right_dims, qhape=right_qims,
                                    charge=charge, dirs=neg_right_dirs)
        ncon_list = tensor_list + [v]
        Av = ncon(ncon_list, matvec_index_list, order=matvec_order)
        Av = Av.to_ndarray()
        Av = np.transpose(Av, left_perm)
        Av = np.reshape(Av, (left_flatdim,))
        if print_progress:
            print(".", end='', flush=True)
        return Av

    def rmatvec(v, charge=0):
        v = np.reshape(v, left_flatdims)
        # TODO Taking the conjugate of v twice is the easiest way to get
        # the qhape to be right. It's not the fastest though, so this
        # should be fixed later.
        v = np.conjugate(v)
        v = commontype.from_ndarray(v, shape=left_dims, qhape=left_qims,
                                    charge=charge, dirs=neg_left_dirs)
        v = v.conjugate()
        ncon_list = tensor_list_conj + [v]
        Av = ncon(ncon_list, rmatvec_index_list, order=rmatvec_order)
        Av = Av.to_ndarray()
        Av = np.transpose(Av, right_perm)
        Av = np.reshape(Av, (right_flatdim,))
        if print_progress:
            print(".", end='', flush=True)
        return Av

    def matmat(v, charge=0):
        d = v.shape[1]
        v = np.reshape(v, right_flatdims+[d])
        if right_qims is not None:
            new_qhape = right_qims+[[charge]]
        else:
            new_qhape = None
        if neg_right_dirs is not None:
            new_dirs = neg_right_dirs+[-1]
        else:
            new_dirs = None
        v = commontype.from_ndarray(v, shape=right_dims+[[d]],
                                    qhape=new_qhape, dirs=new_dirs)
        ncon_list = tensor_list + [v]
        Av = ncon(ncon_list, matmat_index_list, order=matmat_order)
        Av = Av.to_ndarray()
        Av = np.transpose(Av, left_perm+[len(left_perm)])
        Av = np.reshape(Av, (left_flatdim, d))
        if print_progress:
            print(".", end='', flush=True)
        return Av

    if chis is not None:
        n_vals = max(chis)
    elif "k" in kwargs:
        n_vals = kwargs["k"]
        del(kwargs["k"])
    else:
        n_vals = 6
    mindim = min(left_flatdim, right_flatdim)
    if n_vals >= mindim:
        n_vals = mindim -1

    if print_progress:
        print("Diagonalizing...", end="")

    return (matvec, rmatvec, matmat,
            left_qims, left_dims, left_dirs, left_flatdims, left_flatdim,
            right_qims, right_dims, right_dirs, right_flatdims, right_flatdim,
            commontype, commondtype, commonqodulus, n_vals)

# SVD

def ncon_sparsesvd(tensor_list, index_list, matvec_order=None,
                   rmatvec_order=None, matmat_order=None,
                   right_inds=None, left_inds=None, print_progress=False,
                   qnums_do=(), chis=None, eps=0., return_error=False,
                   truncate=True, trunc_err_func=None, norm_sq=None,
                   **kwargs):
    (matvec, rmatvec, matmat,
     left_qims, left_dims, left_dirs, left_flatdims, left_flatdim,
     right_qims, right_dims, right_dirs, right_flatdims, right_flatdim,
     commontype, commondtype, commonqodulus, n_sings) = common_preprocess(
         tensor_list, index_list, matvec_order, rmatvec_order, matmat_order,
         left_inds, right_inds, print_progress=print_progress, chis=chis,
         kwargs=kwargs
    )

    if issubclass(commontype, AbelianTensor):
        # For AbelianTensors.
        # Figure out the list of charges for singular vectors.
        qnums = get_qnums(right_qims, commonqodulus, qnums_do)

        # Initialize S and U.
        S_dtype = np.float_
        S = commontype.empty(shape=[[n_sings]*len(qnums)],
                             qhape=[qnums], invar=False,
                             dirs=[1], dtype=S_dtype)
        U = commontype.empty(shape=left_dims+[[n_sings]*len(qnums)],
                             qhape=left_qims+[qnums],
                             dirs=left_dirs+[-1], dtype=commondtype)
        V = commontype.empty(shape=[[n_sings]*len(qnums)]+right_dims,
                             qhape=[qnums]+right_qims,
                             dirs=[1]+right_dirs, dtype=commondtype)

        # Find the eigenvectors in all the charge sectors one by one.
        for q in qnums:
            U_block, S_block, V_block = get_svdblocks(
                matvec, rmatvec, matmat, q, n_sings,
                left_flatdim, right_flatdim, commondtype, commontype,
                left_qims, left_dims, left_dirs, left_flatdims,
                right_qims, right_dims, right_dirs, right_flatdims,
                **kwargs
            )
            S[(q,)] = S_block
            for k, v in U_block.sects.items():
                U[k] = v
            for k, v in V_block.sects.items():
                V[k] = v

    else:
        # For regular tensors.
        U, S, V = get_svd(matvec, rmatvec, matmat, n_sings,
                          left_dims, right_dims, left_flatdim, right_flatdim,
                          commontype, commondtype, **kwargs)

    if truncate:
        S, U, V, err = truncate_func(S, u=U, v=V, chis=chis, eps=eps,
                                trunc_err_func=trunc_err_func,norm_sq=norm_sq,
                                return_error=True)
    else:
        err = 0.

    if print_progress:
        print()

    retval = (U, S, V)
    if return_error:
        retval += (err,)
    if len(retval) == 1:
        retval = retval[0]
    return retval


def get_svdblocks(matvec, rmatvec, matmat, charge, n_sings, left_flatdim,
                  right_flatdim, commondtype, commontype,
                  left_qims, left_dims, left_dirs, left_flatdims,
                  right_qims, right_dims, right_dirs, right_flatdims,
                  **kwargs):
    lo = spsla.LinearOperator(
        (left_flatdim, right_flatdim),
        matvec=fct.partial(matvec, charge=charge),
        rmatvec=fct.partial(rmatvec, charge=charge),
        matmat=fct.partial(matmat, charge=charge),
        dtype=commondtype
    )
    U_block, S_block, V_block = spsla.svds(lo, k=n_sings, **kwargs)

    order = np.argsort(-np.abs(S_block))
    S_block = S_block[order]
    U_block = U_block[:,order]
    U_block = np.reshape(U_block, left_flatdims+[n_sings])
    U_block = commontype.from_ndarray(U_block, shape=left_dims+[[n_sings]],
                                      qhape=left_qims+[[charge]],
                                      dirs=left_dirs+[-1])
    V_block = V_block[order,:]
    V_block = np.reshape(V_block, [n_sings]+right_flatdims)
    V_block = commontype.from_ndarray(V_block, shape=[[n_sings]]+right_dims,
                                      qhape=[[charge]]+right_qims,
                                      dirs=[1]+right_dirs)
    retval = (U_block, S_block, V_block)
    return retval


def get_svd(matvec, rmatvec, matmat, n_sings, left_dims, right_dims,
            left_flatdim, right_flatdim, commontype, commondtype, **kwargs):
    lo = spsla.LinearOperator((left_flatdim, right_flatdim),
                              matvec=matvec, rmatvec=rmatvec, matmat=matmat,
                              dtype=commondtype)
    U, S, V = spsla.svds(lo, k=n_sings, **kwargs)

    order = np.argsort(-np.abs(S))
    S = S[order]
    S = commontype.from_ndarray(S)
    U = U[:, order]
    U = commontype.from_ndarray(U)
    U = U.reshape(left_dims+[n_sings])
    V = V[order, :]
    V = commontype.from_ndarray(V)
    V = V.reshape([n_sings]+right_dims)
    retval = (U, S, V)
    return retval

# Eig

def ncon_sparseeig(tensor_list, index_list, right_inds, left_inds,
                   matvec_order=None, rmatvec_order=None, matmat_order=None,
                   hermitian=False, print_progress=False, qnums_do=(),
                   return_eigenvectors=True, ncon_func=None, chis=None,
                   eps=0., return_error=False, truncate=True, 
                   trunc_err_func=None, norm_sq=None, **kwargs):
    (matvec, rmatvec, matmat,
     left_qims, left_dims, left_dirs, left_flatdims, left_flatdim,
     right_qims, right_dims, right_dirs, right_flatdims, right_flatdim,
     commontype, commondtype, commonqodulus, n_eigs) = common_preprocess(
         tensor_list, index_list, matvec_order, rmatvec_order, matmat_order,
         left_inds, right_inds, print_progress=print_progress, chis=chis,
         kwargs=kwargs
    )

    if issubclass(commontype, AbelianTensor):
        # For AbelianTensors.
        # Figure out the list of charges for eigenvectors.
        qnums = get_qnums(right_qims, commonqodulus, qnums_do)

        # Initialize S and U.
        S_dtype = np.float_ if hermitian else np.complex_
        S = commontype.empty(shape=[[n_eigs]*len(qnums)],
                             qhape=[qnums], invar=False,
                             dirs=[1], dtype=S_dtype)
        if return_eigenvectors:
            U_dtype = commondtype if hermitian else np.complex_
            U = commontype.empty(shape=left_dims+[[n_eigs]*len(qnums)],
                                 qhape=left_qims+[qnums],
                                 dirs=left_dirs+[-1], dtype=U_dtype)

        # Find the eigenvectors in all the charge sectors one by one.
        for q in qnums:
            blocks = get_eigblocks(
                matvec, q, hermitian, n_eigs, return_eigenvectors,
                right_flatdim, commondtype, commontype, right_qims,
                right_dims, right_dirs, right_flatdims,
                **kwargs
            )
            S[(q,)] = blocks[0]
            if return_eigenvectors:
                U_block = blocks[1]
                for k, v in U_block.sects.items():
                    U[k] = v
    else:
        # For regular tensors.
        res = get_eig(matvec, hermitian, n_eigs, return_eigenvectors,
                      right_dims, right_flatdim, commontype, commondtype,
                      **kwargs)
        S = res[0]
        if return_eigenvectors:
            U = res[1]

    if truncate:
        U = U if return_eigenvectors else None
        res = truncate_func(S, u=U, chis=chis, eps=eps,
                            trunc_err_func=trunc_err_func,norm_sq=norm_sq,
                            return_error=True)
        if return_eigenvectors:
            S, U, err = res
        else:
            S, err = res
    else:
        err = 0.

    if print_progress:
        print()

    retval = (S,)
    if return_eigenvectors:
        retval += (U,)
    if return_error:
        retval += (err,)
    if len(retval) == 1:
        retval = retval[0]
    return retval


def get_eigblocks(matvec, charge, hermitian, n_eigs, return_eigenvectors,
                  right_flatdim, commondtype, commontype, right_qims,
                  right_dims, right_dirs, right_flatdims, **kwargs):
    lo = spsla.LinearOperator(
        (right_flatdim, right_flatdim), fct.partial(matvec, charge=charge),
        dtype=commondtype
    )
    if hermitian:
        res_block = spsla.eigsh(
            lo, return_eigenvectors=return_eigenvectors, k=n_eigs,
            **kwargs
        )
    else:
        res_block = spsla.eigs(
            lo, return_eigenvectors=return_eigenvectors, k=n_eigs,
            **kwargs
        )
    if return_eigenvectors:
        S_block, U_block = res_block
    else:
        S_block = res_block

    order = np.argsort(-np.abs(S_block))
    S_block = S_block[order]
    if return_eigenvectors:
        U_block = U_block[:,order]
        U_block = np.reshape(U_block, right_flatdims+[n_eigs])
        U_block = commontype.from_ndarray(U_block, shape=right_dims+[[n_eigs]],
                                          qhape=right_qims+[[charge]],
                                          dirs=right_dirs+[-1])
    retval = (S_block,)
    if return_eigenvectors:
        retval += (U_block,)
    return retval


def get_eig(matvec, hermitian, n_eigs, return_eigenvectors, right_dims,
            right_flatdim, commontype, commondtype, **kwargs):
    lo = spsla.LinearOperator((right_flatdim, right_flatdim),
                              matvec, dtype=commondtype)
    # DEBUG this shouldn't be necessary, but see
    # https://github.com/opencollab/arpack-ng/issues/79
    #v0 = np.random.rand(right_flatdim, right_flatdim)
    #v0 = lo.matvec(v0)
    # END DEBUG
    if hermitian:
        res = spsla.eigsh(lo, k=n_eigs,
                          return_eigenvectors=return_eigenvectors,
                          #v0=v0,  # DEBUG
                          **kwargs)
    else:
        res = spsla.eigs(lo, k=n_eigs,
                         return_eigenvectors=return_eigenvectors,
                         #v0=v0,  # DEBUG
                         **kwargs)
    if return_eigenvectors:
        S, U = res
        U = commontype.from_ndarray(U)
        U = U.reshape(right_dims+[n_eigs])
    else:
        S = res
    order = np.argsort(-np.abs(S))
    S = S[order]
    S = commontype.from_ndarray(S)
    if return_eigenvectors:
        U = U[...,order]
        U = commontype.from_ndarray(U)
    retval = (S,)
    if return_eigenvectors:
        retval += (U,)
    return retval


