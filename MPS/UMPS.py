import numpy as np
import logging
import warnings
from copy import deepcopy
from ncon import ncon
from tntools.ncon_sparseeig import ncon_sparseeig
from tensors import Tensor

class UMPS:

    min_eps = 1e-10

    copy = deepcopy
    __copy__ = copy

    def __init__(self, tensor, weights, normfactor=1.):
        self.tensor = tensor
        self.weights = weights
        self.canonical = False
        self.normalized = False
        self.normfactor = normfactor
        return None

    @classmethod
    def random(cls, physdim, virtdim, tensorcls=Tensor, dtype=np.complex_,
               random_weights=False):
        dirs = [1,-1,-1]
        tensor = tensorcls.random([virtdim, physdim, virtdim],
                                  dirs=dirs).astype(dtype)-0.5
        if dtype==np.complex_:
            tensor += 1j*(tensorcls.random([virtdim, physdim, virtdim],
                                           dirs=dirs).astype(dtype)-0.5)
        if hasattr(tensor, "defval"):
            tensor.defval = 0  # The 0.5 may have thrown it off.
            tensor.invar = True
        if random_weights:
            weights = tensorcls.random([virtdim], dirs=[1], invar=False)
        else:
            weights = tensorcls.ones([virtdim], dirs=[1], dtype=dtype, invar=False)
        return cls(tensor, weights)

    @staticmethod
    def parse_direction(direction):
        direction = direction.strip().lower()
        if direction in {"l", "left"}:
            return "l"
        elif direction in {"r", "right"}:
            return "r"
        else:
            msg = "Unknown direction: {}".format(direction)
            raise ValueError(msg)

    def reset_normfactor(self):
        normfactor = self.normfactor
        self.normfactor = 1.
        return normfactor

    def virtualdim(self):
        return self.tensortype().flatten_dim(self.tensor.shape[0])

    def tensortype(self):
        return type(self.tensor)

    def conjugate(self):
        conj = self.copy()
        conj.tensor = conj.tensor.conjugate()
        conj.normfactor = np.conjugate(conj.normfactor)
        if conj.normalized:
            conj.__l__ = None if conj.__l__ is None else conj.__l__.conjugate()
            conj.__r__ = None if conj.__r__ is None else conj.__r__.conjugate()
        return conj

    def get_leftweight_tensor(self):
        A = self.tensor.multiply_diag(self.weights, 0, direction="left")
        return A

    def get_rightweight_tensor(self):
        A = self.tensor.multiply_diag(self.weights, 2, direction="right")
        return A

    def get_l(self):
        self.normalize()
        l = self.__l__.copy()
        return l

    def get_r(self):
        self.normalize()
        r = self.__r__.copy()
        return r

    def get_lr(self):
        l = self.get_l()
        r = self.get_r()
        return l, r

    def lr_inner(self, l, r):
        weights = self.weights
        rww = r.multiply_diag(
            weights, 0, direction="left"
        ).multiply_diag(
            weights, 1, direction="right"
        )
        n = ncon((l.conjugate(), rww), ([1,2], [1,2])).value()
        return n

    def transmat(self, direction):
        direction = self.parse_direction(direction)
        if direction == "l":
            return self.transmat_l()
        else:
            return self.transmat_r()

    def apply_transmat(self, x, direction):
        direction = self.parse_direction(direction)
        if direction == "l":
            return self.apply_transmat_l(x)
        else:
            return self.apply_transmat_r(x)

    def transmat_l(self):
        A = self.get_leftweight_tensor()
        A_conj = A.conjugate()
        return ncon((A, A_conj), ([-1,1,-11], [-2,1,-12]))

    def transmat_r(self):
        A = self.get_rightweight_tensor()
        A_conj = A.conjugate()
        return ncon((A, A_conj), ([-1,1,-11], [-2,1,-12]))

    def apply_transmat_l(self, x):
        A = self.get_leftweight_tensor()
        A_conj = A.conjugate()
        return ncon((A_conj, x, A), ([1,2,-1], [1,3], [3,2,-2]))

    def apply_transmat_r(self, x):
        A = self.get_rightweight_tensor()
        A_conj = A.conjugate()
        return ncon((A, x, A_conj), ([-1,2,1], [1,3], [-2,2,3]))

    def transmat_eigs_dense(self, direction, nev):
        direction = self.parse_direction(direction)
        T = self.transmat(direction)
        if direction == "r":
            S, U = T.eig([0,1], [2,3], chis=nev, break_degenerate=True)
        if direction == "l":
            # We take the conjugate of T to match the convention of
            # apply_transmat.
            S, U = T.conjugate().eig([2,3], [0,1], chis=nev,
                                     break_degenerate=True)
        return S, U

    def transmat_eigs_sparse(self, direction, nev):
        # TODO If nev=1, we could use the identity/the old eigenvector
        # as an initial guess for eigs.
        direction = self.parse_direction(direction)
        if direction == "l":
            A = self.get_leftweight_tensor()
            A_conj = A.conjugate()
            S, U = ncon_sparseeig(
                (A_conj, A), ([-1,3,-11], [-2,3,-12]),
                right_inds=[0,1], left_inds=[2,3],
                matvec_order=[1,2,3], rmatvec_order=[11,12,3],
                matmat_order=[1,2,3], chis=[nev]
            )
        else:
            A = self.get_rightweight_tensor()
            A_conj = A.conjugate()
            S, U = ncon_sparseeig(
                (A, A_conj), ([-11,3,-1], [-12,3,-2]),
                right_inds=[0,1], left_inds=[2,3],
                matvec_order=[1,2,3], rmatvec_order=[11,12,3],
                matmat_order=[1,2,3], chis=[nev]
            )
        return S, U

    def transmat_eigs(self, direction, nev, max_dense_virtualdim=30):
        virtualdim = self.virtualdim()
        if virtualdim <= max_dense_virtualdim or nev >= virtualdim**2:
            return self.transmat_eigs_dense(direction, nev)
        else:
            return self.transmat_eigs_sparse(direction, nev)

    def transmat_is_eye(self, t, threshold=1e-8):
        dim = t.shape[0]
        qim = None if t.qhape is None else t.qhape[0]
        eye = type(t).eye(dim, qim=qim)
        eye_norm = self.weights.norm()
        diff = t - eye
        diff_norm = np.sqrt(np.abs(self.lr_inner(diff, diff)))
        t_is_eye = diff_norm/eye_norm < threshold
        return t_is_eye

    def normalize(self, force=False):
        if force or not self.normalized:
            return self.renormalize()
        else:
            return 1.

    def renormalize(self):
        w_norm = self.weights.norm()
        factor = w_norm
        self.weights /= factor
        SL, UL = self.transmat_eigs("l", 1)
        SR, UR = self.transmat_eigs("r", 1)
        # Note that the following relies on the knowledge that since l
        # and r have to be positive semidefinite, they must also have
        # symmetry charge 0.
        S1 = SR.sum()  # Simplest way of getting the single value from SR.
        vect00 = type(SR).ones(SR.shape, qhape=SR.qhape, dirs=SR.dirs,
                               invar=False)
        l = ncon((UL, vect00), ([-1,-2,1], [1]))
        r = ncon((UR, vect00), ([-1,-2,1], [1]))
        if hasattr(l, "invar"):
            l.invar = True
        if hasattr(r, "invar"):
            r.invar = True
        factor *= np.sqrt(S1)
        self.tensor /= np.sqrt(S1)
        # We want both l and r to be Hermitian and pos. semi-def.
        # We know they are that, up to a phase.
        # We can find this phase, and divide it away, because it is also the
        # phase of the trace of l (respectively r).
        r_tr = r.trace()
        phase_r = r_tr/np.abs(r_tr)
        r /= phase_r
        l_tr = l.trace()
        phase_l = l_tr/np.abs(l_tr)
        l /= phase_l
        # Finally divide them by a real scalar that makes
        # their inner product be 1.
        n = self.lr_inner(l, r)
        abs_n = np.abs(n)
        sfac = np.sqrt(abs_n)
        l /= sfac
        r /= sfac
        self.__l__ = l
        self.__r__ = r
        self.normfactor *= factor
        self.normalized = True
        return w_norm

    def canonicalize(self, force=False, **kwargs):
        if force or not self.canonical:
            return self.recanonicalize(**kwargs)
        else:
            return (None, None, None, None, 1, 0)

    def gauge_transform(self, g1i, g2i, return_transformation=False, **kwargs):
        w_old = self.weights
        gwg = ncon((g2i.multiply_diag(w_old, 1, direction="right"), g1i),
                 ([-1,1], [1,-2]))
        if not "eps" in kwargs or kwargs["eps"] < self.min_eps:
            kwargs["eps"] = self.min_eps
        U, w, V, error = gwg.svd(0, 1, return_rel_err=True, **kwargs)

        # Construct the transformations g1 and g2.
        if hasattr(w, "defval"):
            w.defval = np.inf  # TODO Ugly hack to avoid division by zero.
        w_inv = 1/w
        if hasattr(w, "defval"):
            w.defval = 0  # TODO Ugly hack to avoid division by zero.
        g1 = ncon((U.conjugate(), g2i), ([1,-1], [1,-2]))
        g1 = g1.multiply_diag(w_inv, 0, direction="left")
        g1 = g1.multiply_diag(w_old, 1, direction="right")
        g2 = ncon((g1i, V.conjugate()), ([-1,1], [-2,1]))
        g2 = g2.multiply_diag(w_old, 0, direction="left")
        g2 = g2.multiply_diag(w_inv, 1, direction="right")

        T = ncon((g1, self.tensor, g2), ([-1,1], [1,-2,3], [3,-3]))
        self.tensor = T
        self.weights = w
        self.normalized = False
        if return_transformation:
            g1i = ncon((g1i, V.conjugate()), ([-1,1], [-2,1]))
            g2i = ncon((U.conjugate(), g2i), ([1,-1], [1,-2]))
            retval = (g1, g2, g1i, g2i, error)
        else:
            retval = error
        return retval

    # TODO By default, crude should be False.
    def recanonicalize(self, confirm=True, crude=True, confirm_threshold=1e-8,
                       return_transformation=False,**kwargs):
        w_norm = self.normalize()
        l, r = self.get_lr()
        g1i, g2i = self.recanonicalize_build_transformation(l, r)
        g1, g2, g1i, g2i, error = self.gauge_transform(
            g1i, g2i, return_transformation=True, **kwargs
        )
        w_norm *= self.normalize()

        # Check whether canonicality was lost due to truncation.
        if confirm or not crude:
            l, r = self.get_lr()
            l_is_eye = self.transmat_is_eye(l, threshold=confirm_threshold)
            r_is_eye = self.transmat_is_eye(r, threshold=confirm_threshold)
            self.canonical = l_is_eye and r_is_eye
        else:
            self.canonical = False

        if not crude:
            transformation = self.canonicalize(
                eps=0, crude=True, confirm=True,
                return_transformation=return_transformation
            )
            if return_transformation:
                g1b, g2b, g1ib, g2ib, w_normb, errorb = transformation
                if g1b is not None:
                    g1 = ncon((g1b, g1), ([-1,1], [1,-2]))
                if g1ib is not None:
                    g1i = ncon((g1i, g1ib), ([-1,1], [1,-2]))
                if g2b is not None:
                    g2 = ncon((g2, g2b), ([-1,1], [1,-2]))
                if g2ib is not None:
                    g2i = ncon((g2ib, g2i), ([-1,1], [1,-2]))
                w_norm *= w_normb
                error = max(error, errorb)
            if not self.canonical:
                msg = "UMPS.recanonicalize failed to canonicalize."
                warnings.warn(msg)
        if return_transformation:
            retval = (g1, g2, g1i, g2i, w_norm, error)
        else:
            retval = (None, None, None, None, 1, error)
        return retval

    @classmethod
    def recanonicalize_build_transformation(cls, l, r):
        g2i = cls.recanonicalize_build_transformation_l(l)
        g1i = cls.recanonicalize_build_transformation_r(r)
        return g1i, g2i


    @classmethod
    def recanonicalize_build_transformation_l(cls, l):
        evl, Ul = l.eig(0, 1, hermitian=True)
        if (evl/evl.abs().max() < -1e-10).any():
            msg = "Negative values in evl, absing them:\n{}".format(evl)
            warnings.warn(msg)
        evl = evl.abs()
        gi = Ul.conjugate().transpose().multiply_diag(
            evl.sqrt(), 0, direction="left"
        )
        return gi

    @classmethod
    def recanonicalize_build_transformation_r(cls, r):
        evr, Ur = r.eig(0, 1, hermitian=True)
        if (evr/evr.abs().max() < -1e-10).any():
            msg = "Negative values in evr, absing them:\n{}".format(evr)
            warnings.warn(msg)
        evr = evr.abs()
        gi = Ur.multiply_diag(evr.sqrt(), 1, direction="right")
        return gi

    def apply_transmat_l_op(self, O, x):
        l = len(O.shape)
        if l == 2:
            return self.apply_transmat_l_op_onesite(O, x)
        if l == 4:
            return self.apply_transmat_l_op_twosite(O, x)
        if l == 6:
            return self.apply_transmat_l_op_threesite(O, x)
        else:
            msg = "Can't handle an operator with {} legs.".format(l)
            raise ValueError(msg)

    def apply_transmat_r_op(self, O, x):
        l = len(O.shape)
        if l == 2:
            return self.apply_transmat_r_op_onesite(O, x)
        if l == 4:
            return self.apply_transmat_r_op_twosite(O, x)
        if l == 6:
            return self.apply_transmat_r_op_threesite(O, x)
        else:
            msg = "Can't handle an operator with {} legs.".format(l)
            raise ValueError(msg)

    def apply_transmat_l_op_onesite(self, O, x):
        A = self.get_leftweight_tensor()
        A_conj = A.conjugate()
        y = ncon((A_conj,
                  x, O.conjugate(),
                  A),
                 ([3,1,-1],
                  [3,2], [1,4],
                  [2,4,-2]))
        return y

    def apply_transmat_r_op_onesite(self, O, x):
        A = self.get_rightweight_tensor()
        A_conj = A.conjugate()
        y = ncon((A,
                  O, x,
                  A_conj),
                 ([-1,1,3],
                  [1,4], [3,2],
                  [-2,4,2]))
        return y

    def apply_transmat_l_op_twosite(self, O, x):
        A = self.get_leftweight_tensor()
        A_conj = A.conjugate()
        y = ncon((A_conj, A_conj,
                  x, O.conjugate(),
                  A, A),
                 ([1,3,5], [5,6,-1],
                  [1,2], [3,6,4,7],
                  [2,4,8], [8,7,-2]))
        return y

    def apply_transmat_r_op_twosite(self, O, x):
        A = self.get_rightweight_tensor()
        A_conj = A.conjugate()
        y = ncon((A, A,
                  O, x,
                  A_conj, A_conj),
                 ([-1,5,6], [6,3,1],
                  [5,3,7,4], [1,2],
                  [-2,7,8], [8,4,2]))
        return y

    def apply_transmat_l_op_threesite(self, O, x):
        A = self.get_leftweight_tensor()
        A_conj = A.conjugate()
        y = ncon((A_conj, A_conj, A_conj,
                  x, O.conjugate(),
                  A, A, A),
                 ([1,3,5], [5,6,10], [10,9,-1],
                  [1,2], [3,6,9,4,7,12],
                  [2,4,8], [8,7,11], [11,12,-2]))
        return y

    def apply_transmat_r_op_threesite(self, O, x):
        A = self.get_rightweight_tensor()
        A_conj = A.conjugate()
        y = ncon((A, A, A,
                  O, x,
                  A_conj, A_conj, A_conj),
                 ([-1,9,10], [10,5,6], [6,3,1],
                  [9,5,3,11,7,4], [1,2],
                  [-2,11,12], [12,7,8], [8,4,2]))
        return y

    def expect_local(self, O):
        l, r = self.get_lr()
        l = self.apply_transmat_l_op(O, l)
        expectation = self.lr_inner(l, r)
        return expectation

    def expect_twopoint(self, O1, O2, dist):
        local_O1 = self.expect_local(O1)
        local_O2 = self.expect_local(O2)
        disconnected = local_O1 * local_O2
        
        l, r = self.get_lr()
        l = self.apply_transmat_l_op(O1, l)
        r = self.apply_transmat_r_op(O2, r)
        
        result = [self.lr_inner(l, r) - disconnected]
        for i in range(2, dist+1):
            r = self.apply_transmat_r(r)
            value = self.lr_inner(l, r) - disconnected
            result.append(value)
        result = np.array(result)
        return result

    def correlation_length(self):
        if self.virtualdim() < 2:
            xi = 0
        else:
            S, U = self.transmat_eigs("l", 2)
            S0 = S.max()
            S1 = S.min()
            xi = -1/np.log(abs(S1/S0))
        return xi

    def absorb_mpo(self, mpo, is_unitary=False):
        tensor = self.tensor
        weights = self.weights
        tensor = ncon((tensor, mpo), ([-2,1,-22], [-1,1,-21,-11]))
        dirs = None if tensor.dirs is None else [tensor.dirs[0],tensor.dirs[2]]
        tensor = tensor.join_indices([0,1], [3,4], dirs=dirs)
        dirs = weights.dirs
        ones = type(mpo).ones([mpo.shape[0]], dirs=dirs, invar=False)
        # TODO The calling of .diag() is a work around to make this work
        # with AbelianTensor (both have invar=False).
        weights = ncon((weights.diag(), ones.diag()), ([-2,-12], [-1,-11]))
        if weights.dirs is not None:
            dirs = [weights.dirs[1], weights.dirs[3]]
        else:
            dirs = None
        weights = weights.join_indices([0,1], [2,3], dirs=dirs)
        weights = weights.diag().copy()  # The copy avoids a view.
        self.tensor = tensor
        self.weights = weights
        self.canonical = False
        if not is_unitary:
            self.normalized = False
        return None

    def schmidt_profiles_l(self, schmidt_count=np.inf, threshold=1e-5,
                           max_dist=10000):
        # TODO Make this work with symmetries.
        self.canonicalize()
        schmidt_count = min(schmidt_count, self.virtualdim())
        res = {}
        for i in range(schmidt_count):
            for j in range(i):
                A = self.get_leftweight_tensor()
                A_conj = A.conjugate()
                Ai = A[:,:,i]
                Aj = A[:,:,j]
                r = ncon((Ai, Aj.conjugate()), ([-1,1], [-2,1]))
                fids = []
                dist = 0
                while dist < max_dist:
                    dist += 1
                    fid = np.linalg.norm(r, ord='nuc')
                    fids.append(fid)
                    if fid < threshold:
                        break
                    else:
                        r = ncon((A, r, A_conj), ([-1,2,1], [1,3], [-2,2,3]))
                res[(i,j)] = fids
        return res


