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
        self._lag = lag
        self._memlag = memlag
        self._mats = None
        self._coeffs = None
        self._mem = None
        self._gen = None
        self._result = None

    @property
    @abstractmethod
    def mats(self):
        """Each subclass must implement approrpriate method to computes
        correlation matrices"""
        pass

    @property
    def mem(self):
        """Computes the memory matrices from the correlation matrices"""
        if self._mem is None:
            self._mem = memory(self.mats)
        return self._mem

    @property
    def gen(self):
        if self._gen is None:
            self._gen = generator(self.mats, mems=self.mem)
        return self._gen

    @property
    @abstractmethod
    def coeffs(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @property
    def result(self):
        if self._result is None:
            return self.solve()
        return self._result


class ReweightMemory(StatisticMemory):
    """Computes change of measure with memory.

    ...

    Attributes
    ----------
    basis : array-like of ndarray (n_frames, n_basis)
    weights : array-like ndarray (n_frames,)
        Weights for basis
    lag : int
    mem : int, optional
        Number of memory matrices to use. (mem + 1) must evenly divide lag.
    test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis
    """

    def __init__(self, basis, weights, lag, mem=0, test=None):
        super().__init__(lag, memlag=mem)
        self._basis = basis
        self._weights = weights
        self._test = test

    @property
    def mats(self):
        if self._mats is None:
            mats = []
            for t in _memlags(self._lag, self._memlag):
                mats.append(
                    reweight_matrix(
                        self._basis, self._weights, t, test=self._test
                    )
                )
            self._mats = mats
        return self._mats

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = reweight_solve(self.gen)
        return self._coeffs

    def solve(self):
        if self._result is None:
            self._result = reweight_transform(
                self.coeffs, self._basis, self._weights
            )
        return self._result


class ForwardCommittorMemory(StatisticMemory):
    """Computes forward committor with memory.

    ...

    Attributes
    ----------
    basis : array-like of ndarray of float (n_frames, n_basis)
        basis for forward committor, must be 0 outside domain
    weights : array-like ndarray of float (n_frames,)
        Weights for basis
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    lag : int
    mem : int, optional
        Number of memory matrices to use. (mem + 1) must evenly divide lag.
    test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis
    """

    def __init__(
        self, basis, weights, in_domain, guess, lag, mem=0, test=None
    ):
        super().__init__(lag, memlag=mem)
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._guess = guess
        self._test = test

    @property
    def mats(self):
        if self._mats is None:
            mats = []
            for t in _memlags(self._lag, self._memlag):
                mats.append(
                    forward_committor_matrix(
                        self._basis,
                        self._weights,
                        self._in_domain,
                        self._guess,
                        t,
                        test=self._test,
                    )
                )
            self._mats = mats
        return self._mats

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = forward_solve(self.gen)
        return self._coeffs

    def solve(self):
        if self._result is None:
            self._result = forward_transform(
                self.coeffs, self._basis, self._weights, self._guess
            )
        return self._result


class MFPTMemory(StatisticMemory):
    """Computes (forward) mean first passage time with memory.

    ...

    Attributes
    ----------
    basis : array-like of ndarray of float (n_frames, n_basis)
        Basis for MFPT, must be zero outside domain
    weights : array-like ndarray of float (n_frames,)
        Weights for basis
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    lag : int
    mem : int, optional
        Number of memory matrices to use. (mem + 1) must evenly divide lag.
    test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis
    """

    def __init__(
        self, basis, weights, in_domain, guess, lag, mem=0, test=None
    ):
        super().__init__(lag, memlag=mem)
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._guess = guess
        self._test = test

    @property
    def mats(self):
        if self._mats is None:
            mats = []
            for t in _memlags(self._lag, self._memlag):
                mats.append(
                    forward_mfpt_matrix(
                        self._basis,
                        self._weights,
                        self._in_domain,
                        self._guess,
                        t,
                        test=self._test,
                    )
                )
            self._mats = mats
        return self._mats

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = forward_solve(self.gen)
        return self._coeffs

    def solve(self):
        if self._result is None:
            self._result = forward_transform(
                self.coeffs, self._basis, self._weights, self._guess
            )
        return self._result


class BackwardCommittorMemory(StatisticMemory):
    """Computes backward committor with memory.

    ...

    Attributes
    ----------
    w_basis, basis : array-like of ndarray of float (n_frames, n_basis)
        Basis for change-of-measure, and basis for backward committor (must be 0 outside domain)
    weights : array-like ndarray of float (n_frames,)
        Weights for basis
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    lag : int
    mem : int, optional
        Number of memory matrices to use. (mem + 1) must evenly divide lag.
    test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis
    """

    def __init__(
        self, w_basis, basis, weights, in_domain, guess, lag, mem=0, test=None
    ):
        super().__init__(lag, memlag=mem)
        self._w_basis = w_basis
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._guess = guess
        self._test = test

    @property
    def mats(self):
        if self._mats is None:
            mats = []
            for t in _memlags(self._lag, self._memlag):
                mats.append(
                    backward_committor_matrix(
                        self._w_basis,
                        self._basis,
                        self._weights,
                        self._in_domain,
                        self._guess,
                        t,
                        test=self._test,
                    )
                )
            self._mats = mats
        return self._mats

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = backward_solve(self.gen)
        return self._coeffs

    def solve(self):
        if self._result is None:
            self._result = backward_transform(
                self.coeffs,
                self._w_basis,
                self._basis,
                self._weights,
                self._guess,
            )
        return self._result


class MLPTMemory(StatisticMemory):
    """Computes (backward) mean last passage time with memory.

    ...

    Attributes
    ----------
    w_basis, basis : array-like of ndarray of float (n_frames, n_basis)
        Basis for change-of-measure, and basis for MLPT (must be 0 outside domain)
    weights : array-like ndarray of float (n_frames,)
        Weights for basis
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    lag : int
    mem : int, optional
        Number of memory matrices to use. (mem + 1) must evenly divide lag.
    test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis
    """

    def __init__(
        self, w_basis, basis, weights, in_domain, guess, lag, mem=0, test=None
    ):
        super().__init__(lag, memlag=mem)
        self._w_basis = w_basis
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._guess = guess
        self._test = test

    @property
    def mats(self):
        if self._mats is None:
            mats = []
            for t in _memlags(self._lag, self._memlag):
                mats.append(
                    backward_mfpt_matrix(
                        self._w_basis,
                        self._basis,
                        self._weights,
                        self._in_domain,
                        self._guess,
                        t,
                        test=self._test,
                    )
                )
            self._mats = mats
        return self._mats

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = backward_solve(self.gen)
        return self._coeffs

    def solve(self):
        if self._result is None:
            self._result = backward_transform(
                self.coeffs,
                self._w_basis,
                self._basis,
                self._weights,
                self._guess,
            )
        return self._result


class IntegralMemory(StatisticMemory):
    """Class for integral memory statistics. Each subclass
    must implement the mats method which computes the appropriate
    memory matrices.
    """

    def __init__(self, lag, memlag=0):
        super().__init__(lag, memlag=memlag)

    @property
    def coeffs(self):
        raise NotImplementedError("Coefficients not available!")

    def solve(self):
        if self._result is None:
            eye = identity(self.mats, self.mem)
            self._result = integral_solve(self.gen, eye) / (
                self._lag // (self._memlag + 1)
            )
        return self._result


class ReweightIntegralMemory(IntegralMemory):
    """Computes ergodic average of a function with memory.

    ...

    Attributes
    ----------
    basis : array-like of ndarray (n_frames, n_basis)
    weights, values : array-like ndarray (n_frames,)
        Weights for basis, values of function to compute ergodic averages for
    lag : int
    mem : int, optional
        Number of memory matrices to use. (mem + 1) must evenly divide lag.
    test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis
    """

    def __init__(self, basis, weights, values, lag, mem=0, test=None):
        super().__init__(lag, memlag=mem)
        self._basis = basis
        self._weights = weights
        self._values = values
        self._test = test

    @property
    def mats(self):
        if self._mats is None:
            mats = []
            for t in _memlags(self._lag, self._memlag):
                mats.append(
                    reweight_integral_matrix(
                        self._basis,
                        self._weights,
                        self._values,
                        t,
                        test=self._test,
                    )
                )
            self._mats = mats
        return self._mats


class ForwardCommittorIntegralMemory(IntegralMemory):
    """Computes ergodic average of a function with respect to the forward
    committor with memory.

    ...

    Attributes
    ----------
    w_basis, basis : array-like of ndarray of float (n_frames, n_basis)
        Basis for change-of-measure, basis for forward committor (must be 0 outside domain)
    weights, values : array-like ndarray of float (n_frames,)
        Weights for basis, values of function to compute ergodic average for
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    lag : int
    mem : int, optional
        Number of memory matrices to use. (mem + 1) must evenly divide lag.
    w_test, test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis for weights, forward committor
    """

    def __init__(
        self,
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
        super().__init__(lag, memlag=mem)
        self._w_basis = w_basis
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._values = values
        self._guess = guess
        self._w_test = w_test
        self._test = test

    @property
    def mats(self):
        if self._mats is None:
            mats = []
            for t in _memlags(self._lag, self._memlag):
                mats.append(
                    forward_committor_integral_matrix(
                        self._w_basis,
                        self._basis,
                        self._weights,
                        self._in_domain,
                        self._values,
                        self._guess,
                        t,
                        w_test=self._w_test,
                        test=self._test,
                    )
                )
            self._mats = mats
        return self._mats


class MFPTIntegralMemory(IntegralMemory):
    """Computes ergodic average of a function with respect to the (forward)
    MFPT with memory.

    ...

    Attributes
    ----------
    w_basis, basis : array-like of ndarray of float (n_frames, n_basis)
        Basis for change-of-measure, basis for MFPT (must be 0 outside domain)
    weights, values : array-like ndarray of float (n_frames,)
        Weights for basis, values of function to compute ergodic average for
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    lag : int
    mem : int, optional
        Number of memory matrices to use. (mem + 1) must evenly divide lag.
    w_test, test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis for weights, forward committor
    """

    def __init__(
        self,
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
        super().__init__(lag, memlag=mem)
        self._w_basis = w_basis
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._values = values
        self._guess = guess
        self._w_test = w_test
        self._test = test

    @property
    def mats(self):
        if self._mats is None:
            mats = []
            for t in _memlags(self._lag, self._memlag):
                mats.append(
                    forward_mfpt_integral_matrix(
                        self._w_basis,
                        self._basis,
                        self._weights,
                        self._in_domain,
                        self._values,
                        self._guess,
                        t,
                        w_test=self._w_test,
                        test=self._test,
                    )
                )
            self._mats = mats
        return self._mats


class BackwardCommittorIntegralMemory(IntegralMemory):
    """Computes ergodic average of a function with respect to the backward
    committor with memory.

    ...

    Attributes
    ----------
    w_basis, basis : array-like of ndarray of float (n_frames, n_basis)
        Basis for change-of-measure, basis for backward committor (must be 0 outside domain)
    weights, values : array-like ndarray of float (n_frames,)
        Weights for basis, values of function to compute ergodic average for
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    lag : int
    mem : int, optional
        Number of memory matrices to use. (mem + 1) must evenly divide lag.
    w_test, test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis for weights, forward committor
    """

    def __init__(
        self,
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
        super().__init__(lag, memlag=mem)
        self._w_basis = w_basis
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._values = values
        self._guess = guess
        self._w_test = w_test
        self._test = test

    @property
    def mats(self):
        if self._mats is None:
            mats = []
            for t in _memlags(self._lag, self._memlag):
                mats.append(
                    backward_committor_integral_matrix(
                        self._w_basis,
                        self._basis,
                        self._weights,
                        self._in_domain,
                        self._values,
                        self._guess,
                        t,
                        w_test=self._w_test,
                        test=self._test,
                    )
                )
            self._mats = mats
        return self._mats


class MLPTIntegralMemory(IntegralMemory):
    """Computes ergodic average of a function with respect to the (backward)
    MLPT with memory.

    ...

    Attributes
    ----------
    w_basis, basis : array-like of ndarray of float (n_frames, n_basis)
        Basis for change-of-measure, basis for MLPT (must be 0 outside domain)
    weights, values : array-like ndarray of float (n_frames,)
        Weights for basis, values of function to compute ergodic average for
    in_domain : array-like of ndarray of bool (n_frames,)
        Whether each frame is in the domain or not
    guess : array-like of ndarray of float (n_frames,)
        Guess function
    lag : int
    mem : int, optional
        Number of memory matrices to use. (mem + 1) must evenly divide lag.
    w_test, test : optional, array-like of ndarray (n_frames, n_basis)
        Test basis for weights, forward committor
    """

    def __init__(
        self,
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
        super().__init__(lag, memlag=mem)
        self._w_basis = w_basis
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._values = values
        self._guess = guess
        self._w_test = w_test
        self._test = test

    @property
    def mats(self):
        if self._mats is None:
            mats = []
            for t in _memlags(self._lag, self._memlag):
                mats.append(
                    backward_mfpt_integral_matrix(
                        self._w_basis,
                        self._basis,
                        self._weights,
                        self._in_domain,
                        self._values,
                        self._guess,
                        t,
                        w_test=self._w_test,
                        test=self._test,
                    )
                )
            self._mats = mats
        return self._mats


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
    """Compute correlation matrix for forward committors.

    Parameters
    ----------
    basis : array-like (n_trajs,) of ndarray (n_frames, n_basis) of float
        Basis functions; must satisfy boundary conditions
    weights, guess : array-like (n_trajs,) of ndarray (n_frames,) of float
        Weight, guess for committor at each frame in trajectories
    in_domain : array-like (n_trajs,) of ndarray (n_frames,) of bool
        Whether each frame is in the domain (A U B)^c
    test : optional
        Test basis functions; if not specified, will use the same as the right basis

    Returns
    -------
    ndarray (n_basis + 1, n_basis + 1) of float
    """
    if test is None:
        test = basis
    mat = None
    for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess):
        mat = _forward_matrix(x_f, y_f, w, in_d, 0.0, g, lag, mat)
    return _bmat(mat)


def forward_mfpt_matrix(basis, weights, in_domain, guess, lag, test=None):
    """Compute correlation matrix for (forward) mean first passage time."""

    if test is None:
        test = basis
    mat = None
    for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess):
        mat = _forward_matrix(x_f, y_f, w, in_d, 1.0, g, lag, mat)
    return _bmat(mat)


def forward_feynman_kac_matrix(
    basis, weights, in_domain, function, guess, lag, test=None
):
    """Compute correlation matrix for forward Feynman-Kac problem.

    Parameters
    ----------
    basis : array-like (n_trajs,) of ndarray (n_frames, n_basis) of float
        Basis functions; must satisfy boundary conditions
    weights, function, guess : array-like (n_trajs,) of ndarray (n_frames,) of float
        Weight, function to integrate, guess for committor at each frame in trajectories
    in_domain : array-like (n_trajs,) of ndarray (n_frames,) of bool
        Whether each frame is in the domain (A U B)^c
    test : optional
        Test basis functions; if not specified, will use the same as the right basis

    Returns
    -------
    ndarray (2 * n_basis + 1, 2 * n_basis + 1) of float
    """

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
    """Compute correlation matrix for backward committors.

    Parameters
    ----------
    w_basis, basis : array-like (n_trajs,) of ndarray (n_frames, n_basis) of float
        Basis functions to estimate reweighting factor, without boundary
        conditions; basis functions to estimate committor, must satisfy
        boundary conditions
    weights, guess : array-like (n_trajs,) of ndarray (n_frames,) of float
        Weight, guess for committor at each frame in trajectories
    in_domain : array-like (n_trajs,) of ndarray (n_frames,) of bool
        Whether each frame is in the domain (A U B)^c
    test : optional
        Test basis functions; if not specified, will use the same as the right basis

    Returns
    -------
    ndarray (2 * n_basis + 1, 2 * n_basis + 1) of float
    """

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
    """Compute correlation matrix for (backward) mean last passage time."""
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
    """Parameters
    ----------
    w_basis, basis : array-like (n_trajs,) of ndarray (n_frames, n_basis) of float
        Basis functions to estimate reweighting factor, without boundary
        conditions; basis functions to estimate committor, must satisfy
        boundary conditions
    weights, function, guess : array-like (n_trajs,) of ndarray (n_frames,) of float
        Weight, function to integrate, guess for committor at each frame in trajectories
    in_domain : array-like (n_trajs,) of ndarray (n_frames,) of bool
        Whether each frame is in the domain (A U B)^c
    test : optional
        Test basis functions; if not specified, will use the same as the right basis

    Returns
    -------
    ndarray (2 * n_basis + 1, 2 * n_basis + 1) of float
    """
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
    """Compute the correlation matrix for reweighting.

    Parameters
    ----------
    x_w, y_w : ndarray (n_frames, n_basis) of float
        Left and right basis functions.
    w : ndarray (n_frames,) of float
        Initial weights for each frame
    lag : int
    mat : ndarray (2, 2) or None
        Matrix in which to store result
    """
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
    """Compute the correlation matrix for forward-in-time statistics.

    Parameters
    ----------
    x_f, y_f : ndarray (n_frames, n_basis) of float
        Left and right basis functions
    w : ndarray (n_frames,) of float
        Initial weights for each frame
    d_f : ndarray (n_frames,) of bool
        Whether each frame is in the domain
    f_f, g_f : ndarray (n_frames,) of float
        Function to integrate until entrance into domain, guess for statistic.
    lag : int
    mat : ndarray (2, 2) or None
        Matrix in which to store result
    """
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
    """Compute the correlation matrix for backward-in-time statistics.

    Parameters
    ----------
    x_w, y_w : ndarray (n_frames, n_basis) of float
        Left and right basis functions for weights (do not satisfy boundary
        conditions)
    x_b, y_b : ndarray (n_frames, n_basis) of float
        Left and right basis functions for statistic (must satisfy boundary
        conditions)
    w : ndarray (n_frames,) of float
        Initial weights for each frame
    d_b : ndarray (n_frames,) of bool
        Whether each frame is in the domain
    f_b, g_b : ndarray (n_frames,) of float
        Function to integrate until entrance into domain, guess for statistic.
    lag : int
    mat : ndarray (3, 3) or None
        Matrix in which to store result
    """

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
    """Solve problem for forward-in-time statistic.

    Parameters
    ----------
    gen : ndarray of float
        Magic generator operator + memory

    Returns
    -------
    ndarray
        Coefficients
    """
    return np.concatenate([linalg.solve(gen[:-1, :-1], -gen[:-1, -1]), [1.0]])


def _solve_backward(gen):
    """Solve problem for backward-in-time statistic.

    Parameters
    ----------
    gen : ndarray of float
        Magic generator operator + memory

    Returns
    -------
    ndarray
        Coefficients
    """
    return np.concatenate([[1.0], linalg.solve(gen.T[1:, 1:], -gen.T[1:, 0])])


def _solve_integral(gen, eye):
    """Solve problem for integral average statistic.

    Parameters
    ----------
    gen : ndarray of float
        Magic generator operator + memory
    eye : ndarray of float

    Returns
    -------
    ndarray
        Integral result
    """

    mat = linalg.solve(eye, gen)
    forward_coeffs = _solve_forward(mat[1:, 1:])
    backward_coeffs = _solve_backward(mat[:-1, :-1])
    return backward_coeffs @ mat[:-1, 1:] @ forward_coeffs


def _bmat(blocks):
    """Instantiate blocks into a full matrix.

    Parameters
    ----------
    blocks : array-like of 2-D ndarray

    Returns
    -------
    mat : 2-D ndarray
        Full matrix
    """
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
    """Obtain shapes of block matrix

    Returns
    -------
    (tuple of int, tuple of int)
        Dimensions of blocks, where ith index of first tuple and the
        jth index of the second tuple correspond to the row and column
        dimension of the (i, j)th block
    """
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
    """Stride memory lags

    Parameters
    ----------
    lag : int
        Maximum memory lag time.
    mem : int
        Number of evaluations of the memory kernel. Must be one less
        than a memory lag which evenly divides the maximum lag. For example,
        a lag of 32 and number of memory kernels of 3 or 7 is fine (since
        7 + 1 and 3 + 1 evenly divide 32)

    Returns
    -------
    iterable of int
        Lag times at which to evaluate memory kernels.
    """
    assert lag % (mem + 1) == 0
    return np.arange(0, lag + 1, lag // (mem + 1))
