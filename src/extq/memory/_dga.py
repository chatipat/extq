from abc import ABC
from abc import abstractmethod

import numpy as np

from .. import linalg
from . import _matrix
from . import _memory


class StatisticMemory(ABC):
    """Abstract class for memory statistics.

    Each subclass must implement 3 methods:

    * `_compute_cor_mat(self, t)`, which computes the appropriate
      correlation matrix at lag time t,

    * `_compute_coeff(self)`, which solves the problem for coefficients,

    * `_compute_result(self)`, which computes the result.

    """

    def __init__(self, lag, mem=0):
        assert lag % (mem + 1) == 0
        self._lag = lag
        self._mem = mem
        self._delay = lag // (mem + 1)
        self._cor_mats = None
        self._mem_mats = None
        self._coeffs = None
        self._gen = None
        self._result = None

    @abstractmethod
    def _compute_cor_mat(self, t):
        """Compute the correlation matrix at lag time t."""
        pass

    @abstractmethod
    def _compute_coeffs(self):
        """Solve the problem for coefficients."""
        pass

    @abstractmethod
    def _compute_result(self):
        """Compute the result."""
        pass

    @property
    def cor_mats(self):
        """Compute correlation matrices."""
        if self._cor_mats is None:
            self._cor_mats = [
                self._compute_cor_mat(t)
                for t in _memlags(self._lag, self._mem)
            ]
        return self._cor_mats

    @property
    def mem_mats(self):
        """Compute memory matrices from correlation matrices."""
        if self._mem_mats is None:
            self._mem_mats = _memory.memory(self._mem_mats)
        return self._mem_mats

    @property
    def gen(self):
        if self._gen is None:
            self._gen = (
                _memory.generator(self.cor_mats, mems=self.mem_mats)
                / self._delay
            )
        return self._gen

    @property
    def eye(self):
        if self._eye is None:
            self._eye = _memory.identity(self.cor_mats, mems=self.mem_mats)
        return self._eye

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = self._compute_coeffs()
        return self._coeffs

    @property
    def result(self):
        if self._result is None:
            self._result = self._compute_result()
        return self._result


class ReweightMemory(StatisticMemory):
    def __init__(self, basis, weights, lag, mem=0, test=None):
        """Compute the change of measure using DGA with memory.

        Parameters
        ----------
        basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix}
            Basis of trial functions. Must not contain the constant
            function.
        weights : sequence of (n_frames[i],) ndarray
            Weight of the length `lag` short trajectory starting at each
            frame of the trajectories. Note that the last `lag` frames
            of each trajectory must have zero weight, i.e.,
            `weights[i][-lag:] == 0`.
        lag : int
            Maximum lag time, in units of frames.
        mem : int, optional
            Number of memory matrices to use. Note that `mem+1` must
            evenly divide `lag`.
        test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix}, optional
            Basis of test functions against which to minimize the error.
            Must have the same dimension as `basis` and not contain the
            constant function. If None, use `basis`.

        """
        super().__init__(lag, mem=mem)
        self._basis = basis
        self._weights = weights
        self._test = test

    def _compute_cor_mat(self, t):
        return _matrix.reweight_matrix(
            self._basis, self._weights, t, test=self._test
        )

    def _compute_coeffs(self):
        return reweight_solve(self.gen)

    def _compute_result(self):
        return reweight_transform(self.coeffs, self._basis, self._weights)


class ForwardMemory(StatisticMemory):
    def __init__(
        self, basis, weights, in_domain, function, guess, lag, mem=0, test=None
    ):
        super().__init__(lag, mem=mem)
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._function = function
        self._guess = guess
        self._test = test

    def _compute_coeffs(self):
        return forward_solve(self.gen)

    def _compute_result(self):
        return forward_transform(
            self.coeffs, self._basis, self._in_domain, self._guess
        )


class ForwardCommittorMemory(ForwardMemory):
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
        super().__init__(
            basis, weights, in_domain, None, guess, lag, mem=mem, test=test
        )

    def _compute_corr_mat(self, t):
        return _matrix.forward_committor_matrix(
            self._basis,
            self._weights,
            self._in_domain,
            self._guess,
            t,
            test=self._test,
        )


class MFPTMemory(ForwardMemory):
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
        super().__init__(
            basis, weights, in_domain, None, guess, lag, mem=mem, test=test
        )

    def _compute_cor_mat(self, t):
        return _matrix.forward_mfpt_matrix(
            self._basis,
            self._weights,
            self._in_domain,
            self._guess,
            t,
            test=self._test,
        )


class BackwardMemory(StatisticMemory):
    def __init__(
        self,
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
        super().__init__(lag, mem=mem)
        self._w_basis = w_basis
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._function = function
        self._guess = guess
        self._w_test = w_test
        self._test = test

    def _compute_coeffs(self):
        return backward_solve(self.gen)

    def _compute_result(self):
        return backward_transform(
            self.coeffs,
            self._w_basis,
            self._basis,
            self._in_domain,
            self._guess,
        )


class BackwardCommittorMemory(BackwardMemory):
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
        self,
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
        super().__init__(
            w_basis,
            basis,
            weights,
            in_domain,
            None,
            guess,
            lag,
            mem=mem,
            w_test=w_test,
            test=test,
        )

    def _compute_cor_mat(self, t):
        return _matrix.backward_committor_matrix(
            self._w_basis,
            self._basis,
            self._weights,
            self._in_domain,
            self._guess,
            t,
            w_test=self._w_test,
            test=self._test,
        )


class MLPTMemory(BackwardMemory):
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
        self,
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
        super().__init__(
            w_basis,
            basis,
            weights,
            in_domain,
            None,
            guess,
            lag,
            mem=mem,
            w_test=w_test,
            test=test,
        )

    def _compute_cor_mat(self, t):
        return _matrix.backward_mfpt_matrix(
            self._w_basis,
            self._basis,
            self._weights,
            self._in_domain,
            self._guess,
            t,
            w_test=self._w_test,
            test=self._test,
        )


class IntegralMemory(StatisticMemory):
    """Class for integral memory statistics. Each subclass
    must implement the mats method which computes the appropriate
    memory matrices.
    """

    def __init__(self, lag, mem=0):
        super().__init__(lag, mem=mem)

    def _compute_coeffs(self):
        return integral_solve(self.gen, self.eye)

    def _compute_result(self):
        return self._coeffs


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
        super().__init__(lag, mem=mem)
        self._basis = basis
        self._weights = weights
        self._values = values
        self._test = test

    def _compute_cor_mat(self, t):
        return _matrix.reweight_integral_matrix(
            self._basis,
            self._weights,
            self._values,
            t,
            test=self._test,
        )


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
        super().__init__(lag, mem=mem)
        self._w_basis = w_basis
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._values = values
        self._guess = guess
        self._w_test = w_test
        self._test = test

    def _compute_cor_mat(self, t):
        return _matrix.forward_committor_integral_matrix(
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
        super().__init__(lag, mem=mem)
        self._w_basis = w_basis
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._values = values
        self._guess = guess
        self._w_test = w_test
        self._test = test

    def _compute_cor_mat(self, t):
        return _matrix.forward_mfpt_integral_matrix(
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
        super().__init__(lag, mem=mem)
        self._w_basis = w_basis
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._values = values
        self._guess = guess
        self._w_test = w_test
        self._test = test

    def _compute_cor_mat(self, t):
        return _matrix.backward_committor_integral_matrix(
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
        super().__init__(lag, mem=mem)
        self._w_basis = w_basis
        self._basis = basis
        self._weights = weights
        self._in_domain = in_domain
        self._values = values
        self._guess = guess
        self._w_test = w_test
        self._test = test

    def _compute_cor_mat(self, t):
        return _matrix.backward_mfpt_integral_matrix(
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


def reweight(basis, weights, lag, mem=0, test=None):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(_matrix.reweight_matrix(basis, weights, t, test=test))
    return _reweight(mats, basis, weights)


def _reweight(mats, basis, weights):
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
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
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
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
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
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
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
    eye = _memory.identity(mats, mems)
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
    forward_coeffs = np.concatenate(
        [linalg.solve(mat[1:-1, 1:-1], -mat[1:-1, -1]), [1.0]]
    )
    backward_coeffs = np.concatenate(
        [[1.0], linalg.solve(mat.T[1:-1, 1:-1], -mat.T[1:-1, 0])]
    )
    return backward_coeffs @ mat[:-1, 1:] @ forward_coeffs


def _memlags(lag, mem):
    """Returns the lag times at which to evaluate correlation matrices.

    This function acts similarly to `numpy.linspace(0, lag, mem+2)`.

    Parameters
    ----------
    lag : int
        Maximum lag time.
    mem : int
        Number of memory matrices, which are evaluated at equally spaced
        times between time 0 and time `lag`. `mem+1` must evenly divide
        `lag`. For example, with a `lag=32`, `mem=3` and `mem=7` are
        fine since 7+1=8 and 3+1=4 evenly divide 32.

    Returns
    -------
    iterable of int
        Lag times at which to evaluate correlation matrices.

    """
    assert lag % (mem + 1) == 0
    return np.arange(0, lag + 1, lag // (mem + 1))
