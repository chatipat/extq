from abc import ABC, abstractmethod

import numpy as np

from .. import linalg
from ._kernel import backward_integral_kernel
from ._kernel import backward_kernel
from ._kernel import forward_integral_kernel
from ._kernel import forward_kernel
from ._kernel import integral_kernel
from ._kernel import reweight_integral_kernel
from ._kernel import reweight_kernel
from ._memory import generator
from ._memory import identity
from ._memory import memory
from ._tlcc import wtlcc_dense as _build


class StatisticMemory(ABC):
    """Abstract class for memory statistics. Each subclass
    must implement the mats method which computes the appropriate
    memory matrices, as well as the coeffs method which calls
    the appropriate solve function.
    """

    def __init__(self, lag, memlag=0):
        self.lag = lag
        self.memlag = memlag
        self.mats = None
        self.coeffs = None
        self.mem = None
        self.gen = None
        self.result = None

    @abstractmethod
    def mats(self):
        pass

    @property
    def mem(self):
        if self.mem is None:
            self.mem = memory(self.mats)
        return self.mem

    @property
    def gen(self):
        if self.gen is None:
            self.gen = generator(self.mats, self.mem)
        return self.gen

    @abstractmethod
    def coeffs(self):
        pass

    @abstractmethod
    def solve(self):
        pass


class IntegralMemory(StatisticMemory):
    """Class for integral memory statistics. Each subclass
    must implement the mats method which computes the appropriate
    memory matrices, as well as the coeffs method which calls
    the appropriate solve function.
    """

    def __init__(self, lag, memlag=0):
        super().__init__(lag, memlag=memlag)

    def coeffs(self):
        raise NotImplementedError("Coefficients not available!")

    def solve(self):
        if self.result is None:
            eye = identity(mats, mems)
            self.result = integral_solve(gen, eye) / (lag // (memlag + 1))
        return self.result


class ReweightMemory(StatisticMemory):
    """Computes change of measure with memory.

    ...

    Attributes
    ----------
    basis : array-like of ndarray (n_frames, n_basis)
    weights : array-like ndarray (n_frames,)
        Weights for basis
    test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis
    """

    def __init__(self, basis, weights, lag, mem=0, test=None):
        super().__init__(lag, memlag=mem)
        self.basis = basis
        self.weights = weights
        self.test = test

    def mats(self):
        if self.mats is None:
            mats = []
            for t in _memlags(self.lag, self.mem):
                mats.append(
                    reweight_matrix(
                        self.basis, self.weights, t, test=self.test
                    )
                )
            self.mats = mats
        return self.mats

    def coeffs(self):
        if self.coeffs is None:
            self.coeffs = reweight_solve(self.gen)
        return self.coeffs

    def solve(self):
        if self.result is None:
            self.result = reweight_transform(
                self.coeffs, self.basis, self.weights
            )
        return self.result


class ForwardCommittorMemory(StatisticMemory):
    """Computes forward committor with memory.

    ...

    Attributes
    ----------
    basis : array-like of ndarray of float (n_frames, n_basis)
    weights : array-like ndarray of float (n_frames,)
        Weights for basis
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis
    """

    def __init__(
        self, basis, weights, in_domain, guess, lag, mem=0, test=None
    ):
        super().__init__(lag, memlag=mem)
        self.basis = basis
        self.weights = weights
        self.in_domain = in_domain
        self.guess = guess
        self.test = test

    def mats(self):
        if self.mats is None:
            mats = []
            for t in _memlags(self.lag, self.mem):
                mats.append(
                    forward_committor_matrix(
                        self.basis,
                        self.weights,
                        self.in_domain,
                        self.guess,
                        t,
                        test=self.test,
                    )
                )
            self.mats = mats
        return self.mats

    def coeffs(self):
        if self.coeffs is None:
            self.coeffs = forward_solve(self.gen)
        return self.coeffs

    def solve(self):
        if self.result is None:
            self.result = forward_transform(
                self.coeffs, self.basis, self.weights
            )
        return self.result


class MFPTMemory(StatisticMemory):
    """Computes (forward) mean first passage time with memory.

    ...

    Attributes
    ----------
    basis : array-like of ndarray of float (n_frames, n_basis)
    weights : array-like ndarray of float (n_frames,)
        Weights for basis
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis
    """

    def __init__(
        self, basis, weights, in_domain, guess, lag, mem=0, test=None
    ):
        super().__init__(lag, memlag=mem)
        self.basis = basis
        self.weights = weights
        self.in_domain = in_domain
        self.guess = guess
        self.test = test

    def mats(self):
        if self.mats is None:
            mats = []
            for t in _memlags(self.lag, self.mem):
                mats.append(
                    forward_mfpt_matrix(
                        self.basis,
                        self.weights,
                        self.in_domain,
                        self.guess,
                        t,
                        test=self.test,
                    )
                )
            self.mats = mats
        return self.mats

    def coeffs(self):
        if self.coeffs is None:
            self.coeffs = forward_solve(self.gen)
        return self.coeffs

    def solve(self):
        if self.result is None:
            self.result = forward_transform(
                self.coeffs, self.basis, self.weights
            )
        return self.result


class BackwardCommittorMemory(StatisticMemory):
    """Computes backward committor with memory.

    ...

    Attributes
    ----------
    w_basis : array-like of ndarray of float (n_frames, n_basis)
        Basis for change-of-measure
    basis : array-like of ndarray of float (n_frames, n_basis)
    weights : array-like ndarray of float (n_frames,)
        Weights for basis
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis
    """

    def __init__(
        self, w_basis, basis, weights, in_domain, guess, lag, mem=0, test=None
    ):
        super().__init__(lag, memlag=mem)
        self.w_basis = w_basis
        self.basis = basis
        self.weights = weights
        self.in_domain = in_domain
        self.guess = guess
        self.test = test

    def mats(self):
        if self.mats is None:
            mats = []
            for t in _memlags(self.lag, self.mem):
                mats.append(
                    backward_committor_matrix(
                        self.w_basis,
                        self.basis,
                        self.weights,
                        self.in_domain,
                        self.guess,
                        t,
                        test=self.test,
                    )
                )
            self.mats = mats
        return self.mats

    def coeffs(self):
        if self.coeffs is None:
            self.coeffs = backward_solve(self.gen)
        return self.coeffs

    def solve(self):
        if self.result is None:
            self.result = backward_transform(
                self.coeffs, self.w_basis, self.basis, self.weights
            )
        return self.result


class MLPTMemory(StatisticMemory):
    """Computes (backward) mean last passage time with memory.

    ...

    Attributes
    ----------
    w_basis : array-like of ndarray of float (n_frames, n_basis) Basis for change-of-measure basis : array-like of ndarray of float (n_frames, n_basis)
    weights : array-like ndarray of float (n_frames,)
        Weights for basis
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis
    """

    def __init__(
        self, w_basis, basis, weights, in_domain, guess, lag, mem=0, test=None
    ):
        super().__init__(lag, memlag=mem)
        self.w_basis = w_basis
        self.basis = basis
        self.weights = weights
        self.in_domain = in_domain
        self.guess = guess
        self.test = test

    def mats(self):
        if self.mats is None:
            mats = []
            for t in _memlags(self.lag, self.mem):
                mats.append(
                    backward_mfpt_matrix(
                        self.w_basis,
                        self.basis,
                        self.weights,
                        self.in_domain,
                        self.guess,
                        t,
                        test=self.test,
                    )
                )
            self.mats = mats
        return self.mats

    def coeffs(self):
        if self.coeffs is None:
            self.coeffs = backward_solve(self.gen)
        return self.coeffs

    def solve(self):
        if self.result is None:
            self.result = backward_transform(
                self.coeffs, self.w_basis, self.basis, self.weights
            )
        return self.result


def reweight(basis, weights, lag, mem=0, test=None):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(reweight_matrix(basis, weights, t, test=test))
    return _reweight(mats, basis, weights)


def _reweight(mats, basis, weights):
    mems = memory(mats)
    gen = generator(mats, mems)
    coeffs = reweight_solve(gen)
    return reweight_transform(coeffs, basis, weights)


def forward_committor(basis, weights, in_domain, guess, lag, mem=0, test=None):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            forward_committor_matrix(
                basis, weights, in_domain, guess, t, test=test
            )
        )
    return _forward(mats, basis, in_domain, guess)


def forward_mfpt(basis, weights, in_domain, guess, lag, mem=0, test=None):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            forward_mfpt_matrix(basis, weights, in_domain, guess, t, test=test)
        )
    return _forward(mats, basis, in_domain, guess)


def forward_feynman_kac(
    basis, weights, in_domain, function, guess, lag, mem=0, test=None
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            forward_feynman_kac_matrix(
                basis, weights, in_domain, function, guess, t, test=test
            )
        )
    return _forward(mats, basis, in_domain, guess)


def _forward(mats, basis, in_domain, guess):
    mems = memory(mats)
    gen = generator(mats, mems)
    coeffs = forward_solve(gen)
    return forward_transform(coeffs, basis, in_domain, guess)


def backward_committor(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            backward_committor_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _backward(mats, w_basis, basis, in_domain, guess)


def backward_mfpt(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            backward_mfpt_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _backward(mats, w_basis, basis, in_domain, guess)


def backward_feynman_kac(
    w_basis,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            backward_feynman_kac_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                function,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _backward(mats, w_basis, basis, in_domain, guess)


def _backward(mats, w_basis, basis, in_domain, guess):
    mems = memory(mats)
    gen = generator(mats, mems)
    coeffs = backward_solve(gen)
    return backward_transform(coeffs, w_basis, basis, in_domain, guess)


def reweight_integral(basis, weights, values, lag, mem=0, test=None):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            reweight_integral_matrix(basis, weights, values, t, test=test)
        )
    return _integral(mats, lag, mem)


def forward_committor_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            forward_committor_integral_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                values,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _integral(mats, lag, mem)


def forward_mfpt_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            forward_mfpt_integral_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                values,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _integral(mats, lag, mem)


def forward_feynman_kac_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            forward_feynman_kac_integral_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                values,
                function,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _integral(mats, lag, mem)


def backward_committor_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        backward_committor_integral_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            values,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
    return _integral(mats, lag, mem)


def backward_mfpt_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        backward_mfpt_integral_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            values,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
    return _integral(mats, lag, mem)


def backward_feynman_kac_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        backward_feynman_kac_integral_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            values,
            function,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
    return _integral(mats, lag, mem)


def tpt_integral(
    w_basis,
    b_basis,
    f_basis,
    weights,
    in_domain,
    values,
    b_guess,
    f_guess,
    lag,
    mem=0,
    w_test=None,
    b_test=None,
    f_test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            tpt_integral_matrix(
                w_basis,
                b_basis,
                f_basis,
                weights,
                in_domain,
                values,
                b_guess,
                f_guess,
                t,
                w_test=w_test,
                b_test=b_test,
                f_test=f_test,
            )
        )
    return _integral(mats, lag, mem)


def integral(
    w_basis,
    b_basis,
    f_basis,
    weights,
    b_domain,
    f_domain,
    values,
    b_function,
    f_function,
    b_guess,
    f_guess,
    lag,
    mem=0,
    w_test=None,
    b_test=None,
    f_test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            integral_matrix(
                w_basis,
                b_basis,
                f_basis,
                weights,
                b_domain,
                f_domain,
                values,
                b_function,
                f_function,
                b_guess,
                f_guess,
                t,
                w_test=w_test,
                b_test=b_test,
                f_test=f_test,
            )
        )
    return _integral(mats, lag, mem)


def _integral(mats, lag, mem):
    mems = memory(mats)
    gen = generator(mats, mems)
    eye = identity(mats, mems)
    return integral_solve(gen, eye) / (lag // (mem + 1))


def reweight_matrix(basis, weights, lag, test=None):
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, w in zip(basis, test, weights):
        mat = _reweight_matrix(x_w, y_w, w, lag, mat)
    return _bmat(mat)


def reweight_solve(gen):
    return _solve_backward(gen)


def reweight_transform(coeffs, basis, weights):
    result = []
    for x_w, w in zip(basis, weights):
        result.append(_reweight_transform(coeffs, x_w, w))
    return result


def forward_committor_matrix(basis, weights, in_domain, guess, lag, test=None):
    if test is None:
        test = basis
    mat = None
    for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess):
        mat = _forward_matrix(x_f, y_f, w, in_d, 0.0, g, lag, mat)
    return _bmat(mat)


def forward_mfpt_matrix(basis, weights, in_domain, guess, lag, test=None):
    if test is None:
        test = basis
    mat = None
    for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess):
        mat = _forward_matrix(x_f, y_f, w, in_d, 1.0, g, lag, mat)
    return _bmat(mat)


def forward_feynman_kac_matrix(
    basis, weights, in_domain, function, guess, lag, test=None
):
    if test is None:
        test = basis
    mat = None
    for x_f, y_f, w, in_d, f, g in zip(
        test, basis, weights, in_domain, function, guess
    ):
        mat = _forward_matrix(x_f, y_f, w, in_d, f, g, lag, mat)
    return _bmat(mat)


def forward_solve(gen):
    return _solve_forward(gen)


def forward_transform(coeffs, basis, in_domain, guess):
    result = []
    for y_f, in_d, g in zip(basis, in_domain, guess):
        result.append(_forward_transform(coeffs, y_f, in_d, g))
    return result


def backward_committor_matrix(
    w_basis, basis, weights, in_domain, guess, lag, w_test=None, test=None
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, g_b in zip(
        w_basis, w_test, basis, test, weights, in_domain, guess
    ):
        mat = _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, 0.0, g_b, lag, mat)
    return _bmat(mat)


def backward_mfpt_matrix(
    w_basis, basis, weights, in_domain, guess, lag, w_test=None, test=None
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, g_b in zip(
        w_basis, w_test, basis, test, weights, in_domain, guess
    ):
        mat = _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, 1.0, g_b, lag, mat)
    return _bmat(mat)


def backward_feynman_kac_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, f, g in zip(
        w_basis, w_test, basis, test, weights, in_domain, function, guess
    ):
        mat = _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, f, g, lag, mat)
    return _bmat(mat)


def backward_solve(gen):
    return _solve_backward(gen)


def backward_transform(coeffs, w_basis, basis, in_domain, guess):
    result = []
    for x_w, x_b, in_d, g in zip(w_basis, basis, in_domain, guess):
        result.append(_backward_transform(coeffs, x_w, x_b, in_d, g))
    return result


def reweight_integral_matrix(basis, weights, values, lag, test=None):
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, w, v in zip(basis, test, weights, values):
        mat = _reweight_integral_matrix(x_w, y_w, w, v, lag, mat)
    return _bmat(mat)


def forward_committor_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_f, y_f, w, in_d, v, g in zip(
        w_basis, w_test, test, basis, weights, in_domain, values, guess
    ):
        mat = _forward_integral_matrix(
            x_w, y_w, x_f, y_f, w, in_d, v, 0.0, g, lag, mat
        )
    return _bmat(mat)


def forward_mfpt_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_f, y_f, w, in_d, v, g in zip(
        w_basis, w_test, test, basis, weights, in_domain, values, guess
    ):
        mat = _forward_integral_matrix(
            x_w, y_w, x_f, y_f, w, in_d, v, 1.0, g, lag, mat
        )
    return _bmat(mat)


def forward_feynman_kac_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_f, y_f, w, in_d, v, f, g in zip(
        w_basis,
        w_test,
        test,
        basis,
        weights,
        in_domain,
        values,
        function,
        guess,
    ):
        mat = _forward_integral_matrix(
            x_w, y_w, x_f, y_f, w, in_d, v, f, g, lag, mat
        )
    return _bmat(mat)


def backward_committor_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, v, g in zip(
        w_basis,
        w_test,
        basis,
        test,
        weights,
        in_domain,
        values,
        guess,
    ):
        mat = _backward_integral_matrix(
            x_w, y_w, x_b, y_b, w, in_d, v, 0.0, g, lag, mat
        )
    return _bmat(mat)


def backward_mfpt_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, v, g in zip(
        w_basis,
        w_test,
        basis,
        test,
        weights,
        in_domain,
        values,
        guess,
    ):
        mat = _backward_integral_matrix(
            x_w, y_w, x_b, y_b, w, in_d, v, 1.0, g, lag, mat
        )
    return _bmat(mat)


def backward_feynman_kac_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, v, f, g in zip(
        w_basis,
        w_test,
        basis,
        test,
        weights,
        in_domain,
        function,
        values,
        guess,
    ):
        mat = _backward_integral_matrix(
            x_w, y_w, x_b, y_b, w, in_d, v, f, g, lag, mat
        )
    return _bmat(mat)


def tpt_integral_matrix(
    w_basis,
    b_basis,
    f_basis,
    weights,
    in_domain,
    values,
    b_guess,
    f_guess,
    lag,
    w_test=None,
    b_test=None,
    f_test=None,
):
    if w_test is None:
        w_test = w_basis
    if b_test is None:
        b_test = b_basis
    if f_test is None:
        f_test = f_basis
    mat = None
    for x_w, y_w, x_b, y_b, x_f, y_f, w, in_d, v, g_b, g_f in zip(
        w_basis,
        w_test,
        b_basis,
        b_test,
        f_test,
        f_basis,
        weights,
        in_domain,
        values,
        b_guess,
        f_guess,
    ):
        mat = _integral_matrix(
            x_w,
            y_w,
            x_b,
            y_b,
            x_f,
            y_f,
            w,
            in_d,
            in_d,
            v,
            0.0,
            0.0,
            g_b,
            g_f,
            lag,
            mat,
        )
    return _bmat(mat)


def integral_matrix(
    w_basis,
    b_basis,
    f_basis,
    weights,
    b_domain,
    f_domain,
    values,
    b_function,
    f_function,
    b_guess,
    f_guess,
    lag,
    w_test=None,
    b_test=None,
    f_test=None,
):
    if w_test is None:
        w_test = w_basis
    if b_test is None:
        b_test = b_basis
    if f_test is None:
        f_test = f_basis
    mat = None
    for (
        x_w,
        y_w,
        x_b,
        y_b,
        x_f,
        y_f,
        w,
        d_b,
        d_f,
        v,
        f_b,
        f_f,
        g_b,
        g_f,
    ) in zip(
        w_basis,
        w_test,
        b_basis,
        b_test,
        f_test,
        f_basis,
        weights,
        b_domain,
        f_domain,
        values,
        b_function,
        f_function,
        b_guess,
        f_guess,
    ):
        mat = _integral_matrix(
            x_w,
            y_w,
            x_b,
            y_b,
            x_f,
            y_f,
            w,
            d_b,
            d_f,
            v,
            f_b,
            f_f,
            g_b,
            g_f,
            lag,
            mat,
        )
    return _bmat(mat)


def integral_solve(gen, eye):
    return _solve_integral(gen, eye)


def _reweight_matrix(x_w, y_w, w, lag, mat):
    m = reweight_kernel(w, lag)
    if mat is None:
        mat = np.full((2, 2), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    # fmt: on
    return mat


def _reweight_transform(coeffs, x_w, w):
    return w * (coeffs[0] + x_w @ coeffs[1:])


def _forward_matrix(x_f, y_f, w, d_f, f_f, g_f, lag, mat):
    f_f = np.broadcast_to(f_f, len(w) - 1)
    m = forward_kernel(w, d_f, f_f, g_f, lag)
    if mat is None:
        mat = np.full((2, 2), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], x_f , y_f , mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 1], x_f , None, mat[0, 1], lag)
    mat[1, 1] = _build(m[1, 1], None, None, mat[1, 1], lag)
    # fmt: on
    return mat


def _forward_transform(coeffs, y_f, d_f, g_f):
    return g_f + np.where(d_f, y_f @ coeffs[:-1], 0.0) / coeffs[-1]


def _backward_matrix(x_w, y_w, x_b, y_b, w, d_b, f_b, g_b, lag, mat):
    f_b = np.broadcast_to(f_b, len(w) - 1)
    m = backward_kernel(w, d_b, f_b, g_b, lag)
    if mat is None:
        mat = np.full((3, 3), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[0, 2] = _build(m[0, 1], None, y_b , mat[0, 2], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    mat[1, 2] = _build(m[0, 1], x_w , y_b , mat[1, 2], lag)
    mat[2, 2] = _build(m[1, 1], x_b , y_b , mat[2, 2], lag)
    # fmt: on
    return mat


def _backward_transform(coeffs, x_w, x_b, d_b, g_b):
    n = x_w.shape[1] + 1
    com = coeffs[0] + x_w @ coeffs[1:n]
    return g_b + np.where(d_b, x_b @ coeffs[n:], 0.0) / com


def _reweight_integral_matrix(x_w, y_w, w, v, lag, mat):
    m = reweight_integral_kernel(w, v, lag)
    if mat is None:
        mat = np.full((3, 3), None)
    # fmt: off
    # upper left
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    # upper right
    mat[0, 2] = _build(m[0, 1], None, None, mat[0, 2], lag)
    mat[1, 2] = _build(m[0, 1], x_w , None, mat[1, 2], lag)
    # lower right
    mat[2, 2] = _build(m[1, 1], None, None, mat[2, 2], lag)
    # fmt: on
    return mat


def _forward_integral_matrix(
    x_w, y_w, x_f, y_f, w, d_f, v, f_f, g_f, lag, mat
):
    f_f = np.broadcast_to(f_f, len(w) - 1)
    m = forward_integral_kernel(w, d_f, v, f_f, g_f, lag)
    if mat is None:
        mat = np.full((4, 4), None)
    # fmt: off
    # upper left
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    # upper right
    mat[0, 2] = _build(m[0, 1], None, y_f , mat[0, 2], lag)
    mat[0, 3] = _build(m[0, 2], None, None, mat[0, 3], lag)
    mat[1, 2] = _build(m[0, 1], x_w , y_f , mat[1, 2], lag)
    mat[1, 3] = _build(m[0, 2], x_w , None, mat[1, 3], lag)
    # lower right
    mat[2, 2] = _build(m[1, 1], x_f , y_f , mat[2, 2], lag)
    mat[2, 3] = _build(m[1, 2], x_f , None, mat[2, 3], lag)
    mat[3, 3] = _build(m[2, 2], None, None, mat[3, 3], lag)
    # fmt: on
    return mat


def _backward_integral_matrix(
    x_w, y_w, x_b, y_b, w, d_b, v, f_b, g_b, lag, mat
):
    f_b = np.broadcast_to(f_b, len(w) - 1)
    m = backward_integral_kernel(w, d_b, v, f_b, g_b, lag)
    if mat is None:
        mat = np.full((4, 4), None)
    # fmt: off
    # upper left
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[0, 2] = _build(m[0, 1], None, y_b , mat[0, 2], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    mat[1, 2] = _build(m[0, 1], x_w , y_b , mat[1, 2], lag)
    mat[2, 2] = _build(m[1, 1], x_b , y_b , mat[2, 2], lag)
    # upper right
    mat[0, 3] = _build(m[0, 2], None, None, mat[0, 3], lag)
    mat[1, 3] = _build(m[0, 2], x_w , None, mat[1, 3], lag)
    mat[2, 3] = _build(m[1, 2], x_b , None, mat[2, 3], lag)
    # lower right
    mat[3, 3] = _build(m[2, 2], None, None, mat[3, 3], lag)
    # fmt: on
    return mat


def _integral_matrix(
    x_w, y_w, x_b, y_b, x_f, y_f, w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag, mat
):
    f_b = np.broadcast_to(f_b, len(w) - 1)
    f_f = np.broadcast_to(f_f, len(w) - 1)
    m = integral_kernel(w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag)
    if mat is None:
        mat = np.full((5, 5), None)
    # fmt: off
    # upper left
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[0, 2] = _build(m[0, 1], None, y_b , mat[0, 2], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    mat[1, 2] = _build(m[0, 1], x_w , y_b , mat[1, 2], lag)
    mat[2, 2] = _build(m[1, 1], x_b , y_b , mat[2, 2], lag)
    # upper right
    mat[0, 3] = _build(m[0, 2], None, y_f , mat[0, 3], lag)
    mat[0, 4] = _build(m[0, 3], None, None, mat[0, 4], lag)
    mat[1, 3] = _build(m[0, 2], x_w , y_f , mat[1, 3], lag)
    mat[1, 4] = _build(m[0, 3], x_w , None, mat[1, 4], lag)
    mat[2, 3] = _build(m[1, 2], x_b , y_f , mat[2, 3], lag)
    mat[2, 4] = _build(m[1, 3], x_b , None, mat[2, 4], lag)
    # lower right
    mat[3, 3] = _build(m[2, 2], x_f , y_f , mat[3, 3], lag)
    mat[3, 4] = _build(m[2, 3], x_f , None, mat[3, 4], lag)
    mat[4, 4] = _build(m[3, 3], None, None, mat[4, 4], lag)
    # fmt: on
    return mat


def _solve_forward(gen):
    return np.concatenate([linalg.solve(gen[:-1, :-1], -gen[:-1, -1]), [1.0]])


def _solve_backward(gen):
    return np.concatenate([[1.0], linalg.solve(gen.T[1:, 1:], -gen.T[1:, 0])])


def _solve_integral(gen, eye):
    mat = linalg.solve(eye, gen)
    forward_coeffs = _solve_forward(mat[1:, 1:])
    backward_coeffs = _solve_backward(mat[:-1, :-1])
    return backward_coeffs @ mat[:-1, 1:] @ forward_coeffs


def _bmat(blocks):
    s0, s1 = _bshape(blocks)
    si = np.cumsum(np.concatenate([[0], s0]))
    sj = np.cumsum(np.concatenate([[0], s1]))
    mat = np.zeros((si[-1], sj[-1]))
    for i in range(len(s0)):
        for j in range(len(s1)):
            if blocks[i, j] is not None:
                mat[si[i] : si[i + 1], sj[j] : sj[j + 1]] = blocks[i, j]
    return mat


def _bshape(blocks):
    br, bc = blocks.shape
    rows = [None] * br
    cols = [None] * bc
    for i in range(br):
        for j in range(bc):
            if blocks[i, j] is not None:
                r, c = blocks[i, j].shape
                if rows[i] is None:
                    rows[i] = r
                if cols[j] is None:
                    cols[j] = c
                assert (rows[i], cols[j]) == (r, c)
    for r in rows:
        assert r is not None
    for c in cols:
        assert c is not None
    return (tuple(rows), tuple(cols))


def _memlags(lag, mem):
    assert lag % (mem + 1) == 0
    return np.arange(0, lag + 1, lag // (mem + 1))
