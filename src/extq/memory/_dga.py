from abc import ABC
from abc import abstractmethod

import numpy as np

from .. import linalg
from . import _matrix
from ._memory import generator
from ._memory import identity
from ._memory import memory


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
                    _matrix.reweight_matrix(
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
                    _matrix.forward_committor_matrix(
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
                    _matrix.forward_mfpt_matrix(
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
                    _matrix.backward_committor_matrix(
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
                    _matrix.backward_mfpt_matrix(
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
                    _matrix.reweight_integral_matrix(
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
                    _matrix.forward_committor_integral_matrix(
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
                    _matrix.forward_mfpt_integral_matrix(
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
                    _matrix.backward_committor_integral_matrix(
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
                    _matrix.backward_mfpt_integral_matrix(
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
        mats.append(_matrix.reweight_matrix(basis, weights, t, test=test))
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
            _matrix.forward_committor_matrix(
                basis, weights, in_domain, guess, t, test=test
            )
        )
    return _forward(mats, basis, in_domain, guess)


def forward_mfpt(basis, weights, in_domain, guess, lag, mem=0, test=None):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_mfpt_matrix(
                basis, weights, in_domain, guess, t, test=test
            )
        )
    return _forward(mats, basis, in_domain, guess)


def forward_feynman_kac(
    basis, weights, in_domain, function, guess, lag, mem=0, test=None
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_feynman_kac_matrix(
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
            _matrix.backward_committor_matrix(
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
            _matrix.backward_mfpt_matrix(
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
            _matrix.backward_feynman_kac_matrix(
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
            _matrix.reweight_integral_matrix(
                basis, weights, values, t, test=test
            )
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
            _matrix.forward_committor_integral_matrix(
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
            _matrix.forward_mfpt_integral_matrix(
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
            _matrix.forward_feynman_kac_integral_matrix(
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
        _matrix.backward_committor_integral_matrix(
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
        _matrix.backward_mfpt_integral_matrix(
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
        _matrix.backward_feynman_kac_integral_matrix(
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
            _matrix.tpt_integral_matrix(
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
            _matrix.integral_matrix(
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


def reweight_solve(gen):
    """Solve problem for change of measure.

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


def reweight_transform(coeffs, basis, weights):
    result = []
    for x_w, w in zip(basis, weights):
        result.append(w * (coeffs[0] + x_w @ coeffs[1:]))
    return result


def forward_solve(gen):
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


def forward_transform(coeffs, basis, in_domain, guess):
    result = []
    for y_f, d_f, g_f in zip(basis, in_domain, guess):
        result.append(g_f + np.where(d_f, y_f @ coeffs[:-1], 0.0) / coeffs[-1])
    return result


def backward_solve(gen):
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


def backward_transform(coeffs, w_basis, basis, in_domain, guess):
    result = []
    for x_w, x_b, d_b, g_b in zip(w_basis, basis, in_domain, guess):
        n = x_w.shape[1] + 1
        com = coeffs[0] + x_w @ coeffs[1:n]
        result.append(g_b + np.where(d_b, x_b @ coeffs[n:], 0.0) / com)
    return result


def integral_solve(gen, eye):
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
    forward_coeffs = forward_solve(mat[1:, 1:])
    backward_coeffs = backward_solve(mat[:-1, :-1])
    return backward_coeffs @ mat[:-1, 1:] @ forward_coeffs


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
