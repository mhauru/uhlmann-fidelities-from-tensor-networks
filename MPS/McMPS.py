import numpy as np
import scipy.linalg as spla
import logging
import warnings
import operator as opr
import functools as fct
from copy import deepcopy
from .UMPS import UMPS
from ncon import ncon
from tntools.ncon_sparseeig import ncon_sparseeig
from tensors import Tensor

class McMPS:
    
    copy = deepcopy
    __copy__ = copy

    def __init__(self, umps, tensors=[], weightss=[], lbound=0, normfactor=1.):
        self.umps = umps
        self.tensors = tensors
        self.weightss = weightss
        self.normfactor = normfactor
        self.canonical = False
        self.normalized = False
        self.lbound = lbound
        self.rbound = lbound+len(tensors)-1
        self.ls = []
        self.rs = []
        self.last_conj = None
        self.conj_ls = []
        self.conj_rs = []
        self.identifier = np.random.rand()
        return None

    def length(self):
        return len(self.tensors)

    def virtualdim(self, i):
        return self.tensortype().flatten_dim(self.weights(i).shape[0])

    def tensortype(self):
        return self.umps.tensortype()

    def tensor(self, i):
        N = self.length()
        if N < 1 or i < self.lbound or i > self.rbound:
            return self.umps.tensor
        else:
            return self.tensors[i-self.lbound]

    def weights(self, i):
        N = self.length()
        if N <= 1 or i < self.lbound or i > self.rbound:
            return self.umps.weights
        else:
            i = int(i-1/2)
            return self.weightss[i-self.lbound]

    def change_identifier(self):
        self.identifier = np.random.rand()
        return None

    def set_tensor(self, tensor, i):
        self.tensors[i-self.lbound] = tensor
        self.change_identifier()
        return None

    def set_weights(self, weights, i):
        i = int(i-1/2)
        self.weightss[i-self.lbound] = weights
        self.change_identifier()
        return None

    def scale_tensor(self, i, factor):
        if i < self.lbound or i > self.rbound:
            msg = "Position {} out of bounds.".format(i)
            raise ValueError(msg)
        tensor = self.tensor(i)
        tensor *= factor
        for j in range(i-self.lbound, len(self.ls)):
            self.ls[j] *= factor**2
        for j in range(self.rbound-i, len(self.rs)):
            self.rs[j] *= factor**2
        self.change_identifier()
        return None

    def scale_weights(self, i, factor):
        if i < self.lbound or i > self.rbound:
            msg = "Position {} out of bounds.".format(i)
            raise ValueError(msg)
        w = self.weights(i)
        w *= factor
        for j in range(int(i+3/2)-self.lbound-1, len(self.ls)):
            self.ls[j] *= factor**2
        for j in range(self.rbound-1-int(i-3/2), len(self.rs)):
            self.rs[j] *= factor**2
        self.change_identifier()
        return None

    def reset_normfactor(self):
        umps_normfactor = self.umps.reset_normfactor()
        normfactor = self.normfactor
        self.normfactor = 1.
        self.change_identifier()
        return normfactor, umps_normfactor

    def conjugate(self):
        conj = self.copy()
        conj.normfactor = np.conjugate(conj.normfactor)
        conj.umps = conj.umps.conjugate()
        for i in range(len(conj.tensors)):
            conj.tensors[i] = conj.tensors[i].conjugate()
        for i in range(len(conj.ls)):
            conj.ls[i] = conj.ls[i].conjugate()
        for i in range(len(conj.rs)):
            conj.rs[i] = conj.rs[i].conjugate()
        conj.last_conj = None
        conj.conj_ls = []
        conj.conj_rs = []
        conj.change_identifier()
        return conj

    def get_leftweight_tensor(self, i):
        tensor = self.tensor(i)
        weights = self.weights(i-1/2)
        A = tensor.multiply_diag(weights, 0, direction="left")
        return A

    def get_rightweight_tensor(self, i):
        tensor = self.tensor(i)
        weights = self.weights(i+1/2)
        A = tensor.multiply_diag(weights, 2, direction="right")
        return A

    def get_lr_umps_mixed(self, conj):
        nev = 1
        AL = self.umps.get_leftweight_tensor()
        BL = conj.umps.get_leftweight_tensor()
        AL_conj = AL.conjugate()
        SL, UL = ncon_sparseeig(
            (AL_conj, BL), ([-1,3,-11], [-2,3,-12]),
            right_inds=[0,1], left_inds=[2,3],
            matvec_order=[1,2,3], rmatvec_order=[11,12,3],
            matmat_order=[1,2,3], chis=[nev]
        )

        AR = self.umps.get_rightweight_tensor()
        BR = conj.umps.get_rightweight_tensor()
        BR_conj = BR.conjugate()
        SR, UR = ncon_sparseeig(
            (AR, BR_conj), ([-11,3,-1], [-12,3,-2]),
            right_inds=[0,1], left_inds=[2,3],
            matvec_order=[1,2,3], rmatvec_order=[11,12,3],
            matmat_order=[1,2,3], chis=[nev]
        )

        l = UL[:,:,0]
        r = UR[:,:,0]
        comb_fact = (self.umps.normfactor * np.conjugate(conj.umps.normfactor))
        L_fact = SL[0]*comb_fact
        R_fact = SR[0]*comb_fact
        if np.abs(L_fact - 1) > 1e-12 or np.abs(R_fact - 1) > 1e-12:
            logging.info("Eigenvalues in get_lr_umps_mixed: {} & {}"
                         .format(L_fact, R_fact))
        weights_top = self.umps.weights
        weights_bottom = conj.umps.weights
        rww = r.multiply_diag(
            weights_top, 0, direction="left"
        ).multiply_diag(
            weights_bottom, 1, direction="right"
        )
        n = ncon((l.conjugate(), rww), ([1,2], [1,2])).value()
        sfac = np.sqrt(n)
        l /= np.conjugate(sfac)
        r /= sfac
        self.conj_ls = [l]
        self.conj_rs = [r]
        self.last_conj = conj.identifier
        return l, r

    def get_l_mixed(self, conj, i):
        # Note that, compared to get_l, we also need to store the l of
        # the mixed UMPS in ls.
        if self.last_conj != conj.identifier:
            self.conj_ls = []
            self.conj_rs = []
            self.last_conj = conj.identifier
        lbound = min(self.lbound, conj.lbound)
        ls_index = 0 if i <= lbound else i - lbound
        if len(self.conj_ls) > ls_index:
            l = self.conj_ls[ls_index].copy()
        else:
            if i <= lbound:
                l, _ = self.get_lr_umps_mixed(conj)
            else:
                largest_index = max(len(self.conj_ls)-1, 0)
                leftmost_pos = largest_index+lbound
                l = self.get_l_mixed(conj, leftmost_pos)
                for j in range(leftmost_pos, i):
                    l = self.apply_transmat_l(l, j, conj_mps=conj)
                    self.conj_ls.append(l.copy())
        return l

    def get_r_mixed(self, conj, i):
        # Note that, compared to get_r, we also need to store the r of
        # the mixed UMPS in rs.
        if self.last_conj != conj.identifier:
            self.conj_ls = []
            self.conj_rs = []
            self.last_conj = conj.identifier
        rbound = max(self.rbound, conj.rbound)
        rs_index = 0 if i >= rbound else rbound - i
        if len(self.conj_rs) > rs_index:
            r = self.conj_rs[rs_index].copy()
        else:
            if i >= rbound:
                _, r = self.get_lr_umps_mixed(conj)
            else:
                largest_index = max(len(self.conj_rs)-1, 0)
                rightmost_pos = -largest_index+rbound
                r = self.get_r_mixed(conj, rightmost_pos)
                for j in reversed(range(i+1, rightmost_pos+1)):
                    r = self.apply_transmat_r(r, j, conj_mps=conj)
                    self.conj_rs.append(r.copy())
        return r

    def get_l(self, i, conj_mps=None):
        if conj_mps is not None:
            return self.get_l_mixed(conj_mps, i)
        if i <= self.lbound:
            l = self.umps.get_l()
        else:
            ls_index = i-self.lbound-1
            if len(self.ls) > ls_index:
                l = self.ls[ls_index].copy()
            else:
                largest_index = len(self.ls)-1
                leftmost_pos = largest_index+1+self.lbound
                l = self.get_l(leftmost_pos)
                for j in range(leftmost_pos, i):
                    l = self.apply_transmat_l(l, j)
                    self.ls.append(l.copy())
        return l

    def get_r(self, i, conj_mps=None):
        if conj_mps is not None:
            return self.get_r_mixed(conj_mps, i)
        if i >= self.rbound:
            r = self.umps.get_r()
        else:
            rs_index = self.rbound-i-1
            if len(self.rs) > rs_index:
                r = self.rs[rs_index].copy()
            else:
                largest_index = len(self.rs)-1
                rightmost_pos = -largest_index-1+self.rbound
                r = self.get_r(rightmost_pos)
                for j in reversed(range(i+1, rightmost_pos+1)):
                    r = self.apply_transmat_r(r, j)
                    self.rs.append(r.copy())
        return r

    def lr_inner(self, l, r, i, conj_mps=None):
        weights_top = self.weights(i)
        if conj_mps is None:
            weights_bottom = weights_top
        else:
            weights_bottom = conj_mps.weights(i)
        rww = r.multiply_diag(
            weights_top, 0, direction="left"
        ).multiply_diag(
            weights_bottom, 1, direction="right"
        )
        n = ncon((l.conjugate(), rww), ([1,2], [1,2])).value()
        return n

    def transmat(self, direction, i, conj_mps=None):
        direction = self.umps.parse_direction(direction)
        if direction == "l":
            return self.transmat_l(i, conj_mps=conj_mps)
        else:
            return self.transmat_r(i, conj_mps=conj_mps)

    def apply_transmat(self, x, direction, i, conj_mps=None):
        direction = self.parse_direction(direction)
        if direction == "l":
            return self.apply_transmat_l(x, i, conj_mps=conj_mps)
        else:
            return self.apply_transmat_r(x, i, conj_mps=conj_mps)

    def transmat_l(self, i, conj_mps=None):
        A = self.get_leftweight_tensor(i)
        A_conj = A.conjugate() if conj_mps is None else conj_mps.get_leftweight_tensor(i).conjugate()
        return ncon((A, A_conj), ([-1,1,-11], [-2,1,-12]))

    def transmat_r(self, i, conj_mps=None):
        A = self.get_rightweight_tensor(i)
        A_conj = A.conjugate() if conj_mps is None else conj_mps.get_rightweight_tensor(i).conjugate()
        return ncon((A, A_conj), ([-1,1,-11], [-2,1,-12]))

    def apply_transmat_l(self, x, i, conj_mps=None):
        A = self.get_leftweight_tensor(i)
        A_conj = A.conjugate()
        A = A if conj_mps is None else conj_mps.get_leftweight_tensor(i)
        return ncon((A_conj, x, A), ([1,2,-1], [1,3], [3,2,-2]))

    def apply_transmat_r(self, x, i, conj_mps=None):
        A = self.get_rightweight_tensor(i)
        A_conj = A.conjugate() if conj_mps is None else conj_mps.get_rightweight_tensor(i).conjugate()
        return ncon((A, x, A_conj), ([-1,2,1], [1,3], [-2,2,3]))

    def transmat_is_eye(self, t, pos, threshold=1e-8):
        dim = t.shape[0]
        qim = None if t.qhape is None else t.qhape[0]
        eye = type(t).eye(dim, qim=qim)
        eye_norm = self.weights(pos).norm()
        diff = t - eye
        diff_norm = np.sqrt(np.abs(self.lr_inner(diff, diff, pos)))
        t_is_eye = diff_norm/eye_norm < threshold
        return t_is_eye

    def normalize(self, force=False):
        if force or not self.normalized:
            self.renormalize()
        return None

    def renormalize(self):
        N = self.length()
        w_norm = self.umps.normalize()
        if N > 0:
            self.tensors[0] *= np.sqrt(w_norm)
            self.tensors[-1] *= np.sqrt(w_norm)
        if N > 0:
            eval_point = (self.rbound + self.lbound)//2  # Arbitrary choice
            l = self.get_l(eval_point+1)
            r = self.get_r(eval_point)
            norm_sq = self.lr_inner(l, r, eval_point+1/2)
            if np.abs(np.imag(norm_sq))/np.abs(norm_sq) > 1e-10:
                msg = "Norm_sq has an imaginary component: {}".format(norm_sq)
                warnings.warn(msg)
            norm_sq = np.real(norm_sq)
            if norm_sq > 0:
                factor = norm_sq**(-1/(2*N))
                for i in range(self.lbound, self.rbound+1):
                    self.scale_tensor(i, factor)
                self.normfactor *= np.sqrt(norm_sq)
        self.normalized = True
        self.change_identifier()
        return None

    def canonicalize(self, force=False, **kwargs):
        if force or not self.canonical:
            if force:
                self.umps.canonicalize(force=force, **kwargs)
            return self.recanonicalize(**kwargs)
        else:
            return 0

    def gauge_transform(self, g1i, g2i, i, transform_transmats=True, **kwargs):
        w_old = self.weights(i)
        if g2i is not None and g1i is not None:
            gwg = ncon((g2i.multiply_diag(w_old, 1, direction="right"), g1i),
                     ([-1,1], [1,-2]))
        elif g2i is not None:
            gwg = g2i.multiply_diag(w_old, 1, direction="right")
        else:
            gwg = g1i.multiply_diag(w_old, 0, direction="left")
        if not "eps" in kwargs or kwargs["eps"] < self.umps.min_eps:
            kwargs["eps"] = self.umps.min_eps
        U, w, V, error = gwg.svd(0, 1, return_rel_err=True, **kwargs)

        # Construct the transformations g1 and g2.
        if hasattr(w, "defval"):
            w.defval = np.inf  # TODO Ugly hack to avoid division by zero.
        w_inv = 1/w
        if hasattr(w, "defval"):
            w.defval = 0  # TODO Ugly hack to avoid division by zero.
        if g2i is not None:
            g1 = ncon((U.conjugate(), g2i), ([1,-1], [1,-2]))
        else:
            g1 = U.conjugate().transpose()
        g1 = g1.multiply_diag(w_inv, 0, direction="left")
        g1 = g1.multiply_diag(w_old, 1, direction="right")
        if g1i is not None:
            g2 = ncon((g1i, V.conjugate()), ([-1,1], [-2,1]))
        else:
            g2 = V.conjugate().transpose()
        g2 = g2.multiply_diag(w_old, 0, direction="left")
        g2 = g2.multiply_diag(w_inv, 1, direction="right")

        T1 = self.tensor(int(i-1/2))
        T2 = self.tensor(int(i+1/2))
        T1 = ncon((T1, g2), ([-1,-2,3], [3,-3]))
        T2 = ncon((g1, T2), ([-1,1], [1,-2,-3]))
        self.set_tensor(T1, int(i-1/2))
        self.set_tensor(T2, int(i+1/2))
        self.set_weights(w, i)

        l_index = int(i-self.lbound-1/2)
        r_index = int(self.rbound-1-i+1/2)
        if transform_transmats:
            if g2 is not None and 0 <= l_index < len(self.ls):
                l = self.ls[l_index]
                l = ncon((g2.conjugate(), l, g2),
                         ([1,-1], [1,2], [2,-2]))
                self.ls[l_index] = l
            if g1 is not None and 0 <= r_index < len(self.rs):
                r = self.rs[r_index]
                r = ncon((g1, r, g1.conjugate()),
                         ([-1,1], [1,2], [-2,2]))
                self.rs[r_index] = r
        else:
            del(self.ls[l_index:])
            del(self.rs[r_index:])

        return error

    def gauge_transform_boundaries(self, g1, g2, w_norm,
                                   transform_transmats=True):
        self.tensors[0] = ncon((g1, self.tensors[0]), ([-1,1], [1,-2,-3]))
        self.scale_tensor(self.lbound, np.sqrt(w_norm))
        self.tensors[-1] = ncon((self.tensors[-1], g2), ([-1,-2,3], [3,-3]))
        self.scale_tensor(self.rbound, np.sqrt(w_norm))
        if transform_transmats:
            if g2 is not None:
                for l_index in range(self.length()-1, len(self.ls)):
                    l = self.ls[l_index]
                    l = ncon((g2.conjugate().transpose(), l, g2),
                             ([-1,1], [1,2], [2,-2]))
                    self.ls[l_index] = l
            if g1 is not None:
                for r_index in range(self.length()-1, len(self.rs)):
                    r = self.rs[r_index]
                    r = ncon((g1, r, g1.conjugate().transpose()),
                             ([-1,1], [1,2], [2,-2]))
                    self.rs[r_index] = r
        else:
            del(self.ls[self.length()-1:])
            del(self.rs[self.length()-1:])
        return None

    # TODO crude=True shouldn't be the default?
    def recanonicalize(self, crude=True, confirm=True, change_threshold=1e-4,
                       **kwargs):
        N = self.length()
        errors = []
        self.normalize()
        if N < 1:
            umps_transform = self.umps.canonicalize(
                return_transformation=True, confirm=confirm, crude=crude,
                **kwargs)
            return umps_transform[-1]
        # First transform the umps part, without any (significant)
        # truncation.
        umps_transform = self.umps.canonicalize(return_transformation=True,
                                                eps=0, crude=False)
        g1, g2 = umps_transform[0], umps_transform[1]
        w_norm = umps_transform[-2]
        if w_norm != 1 or g1 is not None or g2 is not None:
            self.gauge_transform_boundaries(g1, g2, w_norm,
                                            transform_transmats=False)
            errors.append(umps_transform[-1])

        # Build the ls and rs for the whole system at once.
        ls = []
        for il in range(self.lbound, self.rbound+1):
            ls.append(self.get_l(il+1))
        il += 1
        rs = []
        for ir in reversed(range(self.lbound, self.rbound+1)):
            rs.append(self.get_r(ir-1))
        ir -= 1

        # Transform all the sites within the McMPS window.
        i = 0
        for j in range(self.lbound, self.rbound):
            l = ls[i]
            r = rs[N-2-i]
            g1i, g2i = self.umps.recanonicalize_build_transformation(
                l, r
            )
            error = self.gauge_transform(
                g1i, g2i, j+1/2, transform_transmats=False, **kwargs
            )
            i += 1
            errors.append(error)

        # Push the boundary to the right until canonicalization has no
        # effect anymore. Note that throughout we use ls that don't see
        # the effect of the truncations that have been already done.
        l_umps = self.umps.get_l()
        l = ls[-1]
        old_l = 0
        while True:
            diff = l - l_umps
            w_pos = il-1/2
            diff_norm = np.sqrt(np.abs(self.lr_inner(diff, diff, w_pos)))
            l_umps_norm = np.sqrt(np.abs(self.lr_inner(l_umps, l_umps, w_pos)))
            non_canonicality = np.real(diff_norm/l_umps_norm)
            if non_canonicality < change_threshold:
                break
            # Check that l is still changing.
            diff = l - old_l
            diff_norm = diff.norm()
            l_norm = l.norm()
            if np.abs(diff_norm/l_norm) < 1e-7:
                msg = ("Non-canonicality is only at {}, but we break "
                       "expanding to the right since l has stopped "
                       "changing.".format(non_canonicality))
                raise RuntimeError(msg)
            old_l = l
            # Update l, expand, and canonicalize.
            l = self.apply_transmat_l(l, il)
            gi = self.umps.recanonicalize_build_transformation_l(l)
            self.extend_window_right()
            error = self.gauge_transform(
                None, gi, il-1/2, transform_transmats=False, **kwargs
            )
            errors.append(error)
            il += 1

        # Similarly push the left boundary.
        r_umps = self.umps.get_r()
        r = rs[-1]
        old_r = 0
        while True:
            diff = r - r_umps
            w_pos = ir+1/2
            diff_norm = np.sqrt(np.abs(self.lr_inner(diff, diff, w_pos)))
            r_umps_norm = np.sqrt(np.abs(self.lr_inner(r_umps, r_umps, w_pos)))
            non_canonicality = np.real(diff_norm/r_umps_norm)
            if non_canonicality < change_threshold:
                break
            # Check that r is still changing.
            diff = r - old_r
            diff_norm = diff.norm()
            r_norm = r.norm()
            if np.abs(diff_norm/r_norm) < 1e-7:
                msg = ("Non-canonicality is only at {}, but we break "
                       "expanding to the left since r has stopped "
                       "changing.".format(non_canonicality))
                raise RuntimeError(msg)
            old_r = r
            # Update r, expand, and canonicalize.
            r = self.apply_transmat_r(r, ir)
            gi = self.umps.recanonicalize_build_transformation_r(r)
            self.extend_window_left()
            error = self.gauge_transform(
                gi, None, self.lbound+1/2, transform_transmats=False,
                **kwargs
            )
            errors.append(error)
            ir -= 1

        # Finally transform the UMPS as well, this time truncating.
        # TODO We shouldn't have to "force" this, because we know the
        # UMPS is canonical, we just want to truncate it more than it
        # has been so far.
        umps_transform = self.umps.canonicalize(
            force=True, return_transformation=True, confirm=confirm,
            crude=crude, **kwargs
        )
        g1, g2 = umps_transform[0], umps_transform[1]
        w_norm = umps_transform[-2]
        if w_norm is not None and g1 is not None and g2 is not None:
            self.gauge_transform_boundaries(g1, g2, w_norm,
                                            transform_transmats=False)

        error = max(errors) if errors else 0

        # We have no guarantee that canonicality was actually
        # reached, so check, or flag as not canonical.
        if confirm or not crude:
            all_canonical = self.umps.canonical
            for i in range(1, self.length()):
                l_pos = self.lbound+i
                r_pos = self.rbound-i
                l = self.get_l(l_pos)
                r = self.get_r(r_pos)
                threshold = change_threshold*100
                l_is_eye = self.transmat_is_eye(l, l_pos-1/2,
                                                threshold=threshold)
                r_is_eye = self.transmat_is_eye(r, r_pos+1/2,
                                                threshold=threshold)
                all_canonical = all_canonical and l_is_eye and r_is_eye
                if not all_canonical:
                    # We already know we are not canonical, so no
                    # need to check further.
                    break
            self.canonical = all_canonical
        else:
            self.canonical = False
        if error > 0:
            self.normalized = False

        if not crude:
            error_inner = self.canonicalize(crude=True, confirm=True, eps=0)
            error = max(error, error_inner)
            if not self.canonical:
                msg = "McMPS.recanonicalize failed to canonicalize."
                warnings.warn(msg)

        self.change_identifier()
        return error

    def apply_transmat_l_op(self, O, x, i, conj_mps=None):
        l = len(O.shape)
        if l == 2:
            return self.apply_transmat_l_op_onesite(O, x, i, conj_mps=conj_mps)
        if l == 4:
            return self.apply_transmat_l_op_twosite(O, x, i, conj_mps=conj_mps)
        if l == 6:
            return self.apply_transmat_l_op_threesite(O, x, i,
                                                      conj_mps=conj_mps)

    def apply_transmat_r_op(self, O, x, i, conj_mps=None):
        l = len(O.shape)
        if l == 2:
            return self.apply_transmat_r_op_onesite(O, x, i, conj_mps=conj_mps)
        if l == 4:
            return self.apply_transmat_r_op_twosite(O, x, i, conj_mps=conj_mps)
        if l == 6:
            return self.apply_transmat_r_op_threesite(O, x, i,
                                                      conj_mps=conj_mps)

    def apply_transmat_l_op_onesite(self, O, x, i, conj_mps=None):
        A = self.get_leftweight_tensor(i)
        A_conj = A.conjugate()
        A = A if conj_mps is None else conj_mps.get_leftweight_tensor(i)
        y = ncon((A_conj,
                  x, O.conjugate(),
                  A),
                 ([3,1,-1],
                  [3,2], [1,4],
                  [2,4,-2]))
        return y

    def apply_transmat_r_op_onesite(self, O, x, i, conj_mps=None):
        A = self.get_rightweight_tensor(i)
        A_conj = (A.conjugate() if conj_mps is None
                  else conj_mps.get_rightweight_tensor(i).conjugate())
        y = ncon((A,
                  O, x,
                  A_conj),
                 ([-1,1,3],
                  [1,4], [3,2],
                  [-2,4,2]))
        return y

    def apply_transmat_l_op_twosite(self, O, x, i, conj_mps=None):
        A1 = self.get_leftweight_tensor(i)
        A1_conj = A1.conjugate()
        A1 = A1 if conj_mps is None else conj_mps.get_leftweight_tensor(i)
        A2 = self.get_leftweight_tensor(i+1)
        A2_conj = A2.conjugate()
        A2 = A2 if conj_mps is None else conj_mps.get_leftweight_tensor(i+1)
        y = ncon((A1_conj, A2_conj,
                  x, O.conjugate(),
                  A1, A2),
                 ([1,3,5], [5,6,-1],
                  [1,2], [3,6,4,7],
                  [2,4,8], [8,7,-2]))
        return y

    def apply_transmat_l_op_threesite(self, O, x, i, conj_mps=None):
        A1 = self.get_leftweight_tensor(i-1)
        A1_conj = A1.conjugate()
        A1 = A1 if conj_mps is None else conj_mps.get_leftweight_tensor(i-1)
        A2 = self.get_leftweight_tensor(i)
        A2_conj = A2.conjugate()
        A2 = A2 if conj_mps is None else conj_mps.get_leftweight_tensor(i)
        A3 = self.get_leftweight_tensor(i+1)
        A3_conj = A3.conjugate()
        A3 = A3 if conj_mps is None else conj_mps.get_leftweight_tensor(i+1)
        y = ncon((A1_conj, A2_conj, A3_conj,
                  x, O.conjugate(),
                  A1, A2, A3),
                 ([1,3,5], [5,6,9], [9,10,-1],
                  [1,2], [3,6,10,4,7,12],
                  [2,4,8], [8,7,11], [11,12,-2]))
        return y

    def apply_transmat_r_op_twosite(self, O, x, i, conj_mps=None):
        A1 = self.get_rightweight_tensor(i)
        A1_conj = (A1.conjugate() if conj_mps is None
                   else conj_mps.get_rightweight_tensor(i).conjugate())
        A2 = self.get_rightweight_tensor(i+1)
        A2_conj = (A1.conjugate() if conj_mps is None
                   else conj_mps.get_rightweight_tensor(i+1).conjugate())
        y = ncon((A1, A2,
                  O, x,
                  A1_conj, A2_conj),
                 ([-1,5,6], [6,3,1],
                  [5,3,7,4], [1,2],
                  [-2,7,8], [8,4,2]))
        return y

    def apply_transmat_r_op_threesite(self, O, x, i, conj_mps=None):
        A1 = self.get_rightweight_tensor(i-1)
        A1_conj = (A1.conjugate() if conj_mps is None
                   else conj_mps.get_rightweight_tensor(i-1).conjugate())
        A2 = self.get_rightweight_tensor(i)
        A2_conj = (A1.conjugate() if conj_mps is None
                   else conj_mps.get_rightweight_tensor(i).conjugate())
        A3 = self.get_rightweight_tensor(i+1)
        A3_conj = (A1.conjugate() if conj_mps is None
                   else conj_mps.get_rightweight_tensor(i+1).conjugate())
        y = ncon((A1, A2, A3,
                  O, x,
                  A1_conj, A2_conj, A3_conj),
                 ([-1,10,9], [9,5,6], [6,3,1],
                  [10,5,3,12,7,4], [1,2],
                  [-2,12,11], [11,7,8], [8,4,2]))
        return y

    def expect_local(self, O, i, conj_mps=None, normalize=True):
        self.normalize()
        if conj_mps is not None:
            conj_mps.normalize()
        else:
            if not normalize and np.abs(self.umps.normfactor - 1) > 1e-12:
                msg = ("In expect_local, normalize is False, but the UMPS"
                       "normfactor is {}. Proceeding as if it was 1, though."
                       .format(self.umps.normfactor))
                warnings.warn(msg)
        if not normalize:
            c = conj_mps if conj_mps else self
            rbound = max(c.rbound, self.rbound)
            lbound = min(c.lbound, self.lbound)
            K = rbound - lbound + 1
            N = self.length()
            M = c.length()
            self_cumulative = self.normfactor * self.umps.normfactor**(K-N)
            c_cumulative = c.normfactor * c.umps.normfactor**(K-M)
            total_normfactor = self_cumulative * np.conjugate(c_cumulative)
        Osize = int(len(O.shape)/2)
        if Osize==1:
            l = self.get_l(i, conj_mps=conj_mps)
            r = self.get_r(i, conj_mps=conj_mps)
            l = self.apply_transmat_l_op(O, l, i, conj_mps=conj_mps)
            expectation = self.lr_inner(l, r, i+1/2, conj_mps=conj_mps)
        elif Osize==2:
            l = self.get_l(i, conj_mps=conj_mps)
            r = self.get_r(i+1, conj_mps=conj_mps)
            l = self.apply_transmat_l_op(O, l, i, conj_mps=conj_mps)
            expectation = self.lr_inner(l, r, i+3/2, conj_mps=conj_mps)
        elif Osize==3:
            l = self.get_l(i-1, conj_mps=conj_mps)
            r = self.get_r(i+1, conj_mps=conj_mps)
            l = self.apply_transmat_l_op(O, l, i, conj_mps=conj_mps)
            expectation = self.lr_inner(l, r, i+3/2, conj_mps=conj_mps)
        if not normalize:
            expectation *= total_normfactor
        return expectation

    def expect_twopoint(self, O1, O2, i1, i2, conj_mps=None, normalize=True):
        self.normalize()
        if conj_mps is not None:
            conj_mps.normalize()
        else:
            if not normalize and np.abs(self.umps.normfactor - 1) > 1e-12:
                msg = ("In expect_twopoint, normalize is False, but the UMPS"
                       "normfactor is {}. Proceeding as if it was 1, though."
                       .format(self.umps.normfactor))
                warnings.warn(msg)
        if not normalize:
            c = conj_mps if conj_mps else self
            rbound = max(c.rbound, self.rbound)
            lbound = min(c.lbound, self.lbound)
            K = rbound - lbound + 1
            N = self.length()
            M = c.length()
            self_cumulative = self.normfactor * self.umps.normfactor**(K-N)
            c_cumulative = c.normfactor * c.umps.normfactor**(K-M)
            total_normfactor = self_cumulative * np.conjugate(c_cumulative)
        if i1 == i2:
            msg = "In expect_twopoint, operators are on the same site."
            raise ValueError(msg)
        if i1 > i2:
            i1, i2 = i2, i1
            O1, O2 = O2, O1
        local_O1 = self.expect_local(O1, i1, conj_mps=conj_mps,
                                     normalize=normalize)
        local_O2 = self.expect_local(O2, i2, conj_mps=conj_mps,
                                     normalize=normalize)
        disconnected = local_O1 * local_O2
        
        l = self.get_l(i1, conj_mps=conj_mps)
        r = self.get_r(i2, conj_mps=conj_mps)
        l = self.apply_transmat_l_op(O1, l, i1, conj_mps=conj_mps)
        r = self.apply_transmat_r_op(O2, r, i2, conj_mps=conj_mps)
        for i in range(i1+1, i2):
            l = self.apply_transmat_l(l, i, conj_mps=conj_mps)
        result = self.lr_inner(l, r, i2-1/2, conj_mps=conj_mps)
        if not normalize:
            result *= total_normfactor
        result -= disconnected
        return result

    def extend_window_right(self):
        self.tensors.append(self.umps.tensor.copy())
        self.weightss.append(self.umps.weights.copy())
        self.rbound += 1
        self.normfactor *= self.umps.normfactor
        if self.umps.normalized:
            self.rs = [self.umps.get_r()] + self.rs
        else:
            self.rs = []
            self.normalized = False
        self.canonical = self.canonical and self.umps.canonical
        return None

    def extend_window_left(self):
        self.tensors = [self.umps.tensor.copy()] + self.tensors
        self.weightss = [self.umps.weights.copy()] + self.weightss
        self.lbound -= 1
        self.normfactor *= self.umps.normfactor
        if self.umps.normalized:
            self.ls = [self.umps.get_l()] + self.ls
        else:
            self.ls = []
            self.normalized = False
        self.canonical = self.canonical and self.umps.canonical
        return None

    def absorb_mpo(self, mpo, is_unitary=False):
        self.umps.absorb_mpo(mpo, is_unitary=is_unitary)
        for i in range(self.lbound, self.rbound+1):
            tensor = self.tensor(i)
            tensor = ncon((tensor, mpo), ([-2,1,-22], [-1,1,-21,-11]))
            tensor = tensor.join_indices([0,1], [3,4])
            self.set_tensor(tensor, i)
            if i != self.rbound:
                weights = self.weights(i+1/2)
                ones = type(mpo).ones(mpo.shape[0])
                weights = ncon((weights, ones), ([-2], [-1]))
                weights = weights.join_indices([0,1])
                self.set_weights(weights, i+1/2)
        self.ls = []
        self.rs = []
        self.conj_ls = []
        self.conj_rs = []
        self.canonical = False
        if not is_unitary:
            self.normalized = False
        self.change_identifier()
        return None

    def halfsystem_fidelity_l(self, conj_mps, position, normalize=True,
                              return_u=False):
        if np.abs(position+1/2 - int(np.round(position+1/2))) > 1e-14:
            msg = "position is not a half integer: {}".format(position)
            raise ValueError(msg)
        r_self = self.get_r(int(np.round(position-1/2)))
        r_conj = conj_mps.get_r(int(np.round(position-1/2)))
        gi_self = self.umps.recanonicalize_build_transformation_r(
            r_self
        )
        gi_conj = conj_mps.umps.recanonicalize_build_transformation_r(
            r_conj
        )
        l_mixed = self.get_l(int(np.round(position+1/2)), conj_mps=conj_mps)
        w = self.weights(position)
        wc = conj_mps.weights(position).conjugate()
        lww = l_mixed.conjugate().multiply_diag(w, 0, direction="left")
        lww = lww.multiply_diag(wc, 1, direction="right")
        lww = ncon((gi_self, lww, gi_conj.conjugate()),
                   ([1,-1], [1,2], [2,-2]))
        u, s, v = lww.svd(0, 1)
        l_fid = s.sum()
        if not normalize:
            normfactor = self.normfactor * np.conjugate(conj_mps.normfactor)
            l_fid *= normfactor
        if return_u:
            u = ncon((u.conjugate(), v.conjugate()),
                     ([-1,1], [1,-2]))
            return l_fid, u
        else:
            return l_fid

    def halfsystem_fidelity_r(self, conj_mps, position, normalize=True,
                              return_u=False):
        if np.abs(position+1/2 - int(np.round(position+1/2))) > 1e-14:
            msg = "position is not a half integer: {}".format(position)
            raise ValueError(msg)
        l_self = self.get_l(int(np.round(position+1/2)))
        l_conj = conj_mps.get_l(int(np.round(position+1/2)))
        gi_self = self.umps.recanonicalize_build_transformation_l(
            l_self
        )
        gi_conj = conj_mps.umps.recanonicalize_build_transformation_l(
            l_conj
        )
        r_mixed = self.get_r(int(np.round(position-1/2)), conj_mps=conj_mps)
        w = self.weights(position)
        wc = conj_mps.weights(position).conjugate()
        rww = r_mixed.multiply_diag(w, 0, direction="right")
        rww = rww.multiply_diag(wc, 1, direction="left")
        rww = ncon((gi_self, rww, gi_conj.conjugate()),
                   ([-1,1], [1,2], [-2,2]))
        u, s, v = rww.svd(0, 1)
        r_fid = s.sum()
        if not normalize:
            normfactor = self.normfactor * np.conjugate(conj_mps.normfactor)
            r_fid *= normfactor
        if return_u:
            u = ncon((u.conjugate(), v.conjugate()),
                     ([-1,1], [1,-2]))
            return r_fid, u
        else:
            return r_fid

    def halfsystem_fidelity(self, conj_mps, position, **kwargs):
        l_fid = self.halfsystem_fidelity_l(conj_mps, position, **kwargs)
        r_fid = self.halfsystem_fidelity_r(conj_mps, position, **kwargs)
        return l_fid, r_fid

    def reduced_density_matrix(self, pos_l, pos_r, conj_mps=None,
                               normalize=True):
        if normalize:
            self.normalize()
            if conj_mps is not None:
                conj_mps.normalize()
        width = pos_r - pos_l + 1
        l = self.get_l(pos_l, conj_mps=conj_mps).conjugate()
        for i in range(width):
            w = self.weights(pos_l+i-1/2)
            T = self.tensor(pos_l+i)
            wT = T.multiply_diag(w, 0, direction="left")
            if conj_mps is None:
                wT_conj = wT.conjugate()
            else:
                w_conj = conj_mps.weights(pos_l+i-1/2)
                T_conj = conj_mps.tensor(pos_l+i)
                wT_conj = T_conj.multiply_diag(w_conj, 0, direction="left")
                wT_conj = wT_conj.conjugate()
            l_inds = [-j for j in range(1, 2*i+1)]
            l_inds += [1,2]
            l = ncon((l, wT, wT_conj), (l_inds, [1,-100,-200], [2,-101,-201]))
        r = self.get_r(pos_r, conj_mps=conj_mps)
        w = self.weights(pos_r+1/2)
        if conj_mps is None:
            w_conj = w.conjugate()
        else:
            w_conj = conj_mps.weights(pos_l+width-1/2).conjugate()
        rww = r.multiply_diag(w, 0, direction="left")
        rww = rww.multiply_diag(w_conj, 1, direction="right")
        l_inds = [-j for j in range(1, 2*width+1)]
        l_inds += [1,2]
        rho = ncon((l, rww), (l_inds, [1,2]))
        perm = [2*i for i in range(width)] + [2*i+1 for i in range(width)]
        rho = rho.transpose(perm)
        if not normalize:
            rho *= self.normfactor
            if conj_mps is None:
                rho *= self.normfactor.conjugate()
            else:
                rho *= conj_mps.normfactor.conjugate()
        return rho

    def window_fidelity_costphys(self, conj_mps, pos_l, pos_r=None,
                                 normalize=True, log=False):
        pos_r = pos_l if pos_r is None else pos_r
        rho = self.reduced_density_matrix(pos_l, pos_r, normalize=normalize)
        rho_conj = conj_mps.reduced_density_matrix(pos_l, pos_r,
                                                   normalize=normalize)
        width = pos_r - pos_l + 1
        rho = rho.join_indices(list(range(width)),
                               list(range(width,2*width)))
        rho_conj = rho_conj.join_indices(list(range(width)),
                                         list(range(width,2*width)))
        rho = rho.to_ndarray()
        rho_conj = rho_conj.to_ndarray()
        rho_sqrt = spla.sqrtm(rho)
        rho_conj_sqrt = spla.sqrtm(rho_conj)
        M = np.dot(rho_sqrt, rho_conj_sqrt)
        S = np.linalg.svd(M)[1]
        fid = np.sum(S)
        if log:
            logging.info("Window fidelity: {}".format(fid))
        # TODO do we need normfactors here?
        return fid

    def window_fidelity_costvirt(self, conj_mps, pos_l, pos_r=None,
                                 upto=False, normalize=True, log=False):
        pos_r = pos_l if pos_r is None else pos_r
        res = []

        T = self.transmat_l(pos_l, conj_mps=conj_mps)

        l_self = self.get_l(pos_l)
        l_conj = conj_mps.get_l(pos_l)
        gil_self = self.umps.recanonicalize_build_transformation_l(l_self)
        gil_conj = conj_mps.umps.recanonicalize_build_transformation_l(l_conj)
        T = ncon((gil_self, gil_conj.conjugate(), T),
                 ([-1,1], [-2,2], [1,2,-3,-4]))

        for i in range(pos_l+1, pos_r+1):
            Ti = self.transmat_l(i, conj_mps=conj_mps)
            T = ncon((T, Ti), ([-1,-2,1,2], [1,2,-11,-12]))
            if upto or i==pos_r:
                w = self.weights(i+1/2)
                w_conj = conj_mps.weights(i+1/2)
                Tw = T.multiply_diag(w, 2, direction="right")
                Tw = Tw.multiply_diag(w_conj, 3, direction="left")

                r_self = self.get_r(i)
                r_conj = conj_mps.get_r(i)
                gir_self = self.umps.recanonicalize_build_transformation_r(r_self)
                gir_conj = conj_mps.umps.recanonicalize_build_transformation_r(r_conj)

                Tw = ncon((Tw, gir_self, gir_conj.conjugate()),
                         ([-1,-2,3,4], [3,-3], [4,-4]))

                S = Tw.svd([0,2], [1,3])[1]
                fid = S.sum()
                if log:
                    logging.info("Window fidelity up to {}: {}".format(i, fid))
                res.append(fid)
        res = np.array(res)
        if not normalize:
            normfactor = self.normfactor * np.conjugate(conj_mps.normfactor)
            res *= normfactor
        if upto:
            return res
        else:
            return res[0]

    # The default option.
    def window_fidelity(self, conj_mps, pos_l, pos_r=None, upto=False,
                        normalize=True, log=False):
        pos_r = pos_l if pos_r is None else pos_r
        if pos_r - pos_l < 5 and not upto:
            fid = self.window_fidelity_costphys(conj_mps, pos_l, pos_r,
                                                normalize, log)
        else:
            fid = self.window_fidelity_costvirt(conj_mps, pos_l, pos_r,
                                                upto, normalize, log)
        return fid

    def window_fidelity_separate(self, conj_mps, pos_l, pos_r=None,
                                 normalize=True, return_us=False,
                                 max_counter=1000, eps_conv=1e-8,
                                 initial_us=None):
        pos_r = pos_l if pos_r is None else pos_r

        r_self = self.get_r(pos_r)
        r_conj = conj_mps.get_r(pos_r)
        gir_self = self.umps.recanonicalize_build_transformation_r(r_self)
        gir_conj = conj_mps.umps.recanonicalize_build_transformation_r(r_conj)

        l_self = self.get_l(pos_l)
        l_conj = conj_mps.get_l(pos_l)
        gil_self = self.umps.recanonicalize_build_transformation_l(l_self)
        gil_conj = conj_mps.umps.recanonicalize_build_transformation_l(l_conj)

        # Optimize for unitaries at both ends.
        # First find initial guesses for ul and ur.
        ul, ur = (None, None) if initial_us is None else initial_us
        use_initial = (
            ul is not None
            and
            ur is not None
            and
            ul.compatible_indices(self.tensor(pos_l), 0, 0)
            and
            ul.compatible_indices(conj_mps.tensor(pos_l), 1, 0)
            and
            ur.compatible_indices(self.tensor(pos_r), 0, 2)
            and
            ur.compatible_indices(conj_mps.tensor(pos_r), 1, 2)
        )
        if not use_initial:
            fid, ul = self.halfsystem_fidelity_r(
                conj_mps, pos_l-1/2, normalize=normalize, return_u=True
            )
            fid, ur = self.halfsystem_fidelity_l(
                conj_mps, pos_r+1/2, normalize=normalize, return_u=True
            )
        else:
            fid = np.inf
        change = np.inf
        counter = 0

        # Then iteratively optimize.
        while counter < max_counter and change > eps_conv:
            T_rtrace = ur
            for i in reversed(range(pos_l, pos_r+1)):
                T_rtrace = self.apply_transmat_r(T_rtrace, i,
                                                 conj_mps=conj_mps)
            w = self.weights(pos_l-1/2)
            w_conj = conj_mps.weights(pos_l-1/2)
            T_rtrace = T_rtrace.multiply_diag(w, 0, direction="left")
            T_rtrace = T_rtrace.multiply_diag(w_conj, 1, direction="right")
            u, s, v = T_rtrace.svd(0,1)
            ul = ncon((u.conjugate(), v.conjugate()), ([-1,1], [1,-2]))

            T_ltrace = ul.conjugate()
            for i in range(pos_l, pos_r+1):
                T_ltrace = self.apply_transmat_l(T_ltrace, i,
                                                 conj_mps=conj_mps)
            T_ltrace = T_ltrace.conjugate()
            w = self.weights(pos_r+1/2)
            w_conj = conj_mps.weights(pos_r+1/2)
            T_ltrace = T_ltrace.multiply_diag(w, 0, direction="right")
            T_ltrace = T_ltrace.multiply_diag(w_conj, 1, direction="left")
            u, s, v = T_ltrace.svd(0,1)
            ur = ncon((u.conjugate(), v.conjugate()), ([-1,1], [1,-2]))

            old_fid = fid
            fid = s.sum()
            change = np.abs((fid - old_fid)/fid)
        if not normalize:
            normfactor = self.normfactor * np.conjugate(conj_mps.normfactor)
            fid *= normfactor
        if return_us:
            return fid, ul, ur
        else:
            return fid


