import numpy as np

from . import dgastat, linalg, utils

__all__ = [
    "reweight",
    "forward_committor",
    "forward_mfpt",
    "forward_feynman_kac",
    "backward_committor",
    "backward_mfpt",
    "backward_feynman_kac",
    "HankelDGA",
]


def reweight(
    basis, delay, n_delays, maxlag=None, guess=None, test_basis=None, *, normalize=True
):
    """Estimate the change of measure to the invariant distribution.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the change of measure.
    delay : int
        Difference in lag time (in frames) between matrix evaluations.
    n_delays : int
        Number of different lag times to use. Must be an odd integer.
        The maximum lag time is `n_delays*delay`.
        DGA corresponds to `n_delays=1`.
    maxlag : int
        Number of frames at the end of each trajectory that are required
        to have zero weight. This is the maximum lag time the output
        weights can be used with by other methods.
    guess : list of (n_frames[i],) ndarray of float, optional
        Guess for the change of measure. The last maxlag frames of
        each trajectory must be zero.
        If None, use uniform weights (except for the last lag frames).
    test_basis : list of (n_frames[i], n_basis) ndarray of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the change of
        measure. If None, use the basis that is used to estimate the
        change of measure.
    normalize : bool, optional
        If True (default), normalize output to one.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the change of measure at each frame of the
        trajectory.

    """
    if maxlag is None:
        maxlag = delay * n_delays
    assert 0 < delay * n_delays <= maxlag
    if guess is None:
        guess = utils.uniform_weights(basis, maxlag)
    stat = dgastat.StationaryDistribution(basis, guess, test_basis=test_basis)
    algo = HankelDGA(delay, n_delays).fit(stat)
    out = algo.solution(stat)
    if normalize:
        out = utils.normalize_weights(out)
    return out


def forward_committor(
    basis, weights, in_domain, guess, delay, n_delays, test_basis=None
):
    """Estimate the forward committor using Hankel DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the committor. Must obey boundary conditions.
    delay : int
        Difference in lag time (in frames) between matrix evaluations.
    n_delays : int
        Number of different lag times to use. Must be an odd integer.
        The maximum lag time is `n_delays*delay`.
        DGA corresponds to `n_delays=1`.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the committor.
        If None, use the basis that is used to estimate the committor.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the forward committor at each frame of the
        trajectory.

    """
    return forward_feynman_kac(
        basis,
        weights,
        in_domain,
        0.0,
        guess,
        delay,
        n_delays,
        test_basis=test_basis,
    )


def forward_mfpt(basis, weights, in_domain, guess, delay, n_delays, test_basis=None):
    """Estimate the forward mean first passage time using Hankel DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the mean first passage time. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.
    delay : int
        Difference in lag time (in frames) between matrix evaluations.
    n_delays : int
        Number of different lag times to use. Must be an odd integer.
        The maximum lag time is `n_delays*delay`.
        DGA corresponds to `n_delays=1`.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the forward mean first passage time at each frame of
        the trajectory.

    """
    return forward_feynman_kac(
        basis,
        weights,
        in_domain,
        1.0,
        guess,
        delay,
        n_delays,
        test_basis=test_basis,
    )


def forward_feynman_kac(
    basis, weights, in_domain, function, guess, delay, n_delays, test_basis=None
):
    """Solve the forward Feynman-Kac formula using Hankel DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the solution of the Feynman-Kac formula.
        Must be zero outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    function : list of (n_frames[i]-1,) ndarray of float
        Function to integrate. Note that is defined over transitions,
        not frames.
    guess : list of (n_frames[i],) ndarray of float
        Guess of the solution. Must obey boundary conditions.
    delay : int
        Difference in lag time (in frames) between matrix evaluations.
    n_delays : int
        Number of different lag times to use. Must be an odd integer.
        The maximum lag time is `n_delays*delay`.
        DGA corresponds to `n_delays=1`.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the solution of the forward Feynman-Kac formula at
        each frame of the trajectory.

    """
    stat = dgastat.ForwardFeynmanKac(
        basis, weights, in_domain, function, guess, test_basis=test_basis
    )
    algo = HankelDGA(delay, n_delays).fit(stat)
    return algo.solution(stat)


def backward_committor(
    basis, weights, in_domain, guess, delay, n_delays, test_basis=None
):
    """Estimate the backward committor using Hankel DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the committor. Must obey boundary conditions.
    delay : int
        Difference in lag time (in frames) between matrix evaluations.
    n_delays : int
        Number of different lag times to use. Must be an odd integer.
        The maximum lag time is `n_delays*delay`.
        DGA corresponds to `n_delays=1`.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the committor.
        If None, use the basis that is used to estimate the committor.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the backward committor at each frame of the
        trajectory.

    """
    return backward_feynman_kac(
        basis,
        weights,
        in_domain,
        0.0,
        guess,
        delay,
        n_delays,
        test_basis=test_basis,
    )


def backward_mfpt(basis, weights, in_domain, guess, delay, n_delays, test_basis=None):
    """Estimate the backward mean first passage time using Hankel DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the mean first passage time. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.
    delay : int
        Difference in lag time (in frames) between matrix evaluations.
    n_delays : int
        Number of different lag times to use. Must be an odd integer.
        The maximum lag time is `n_delays*delay`.
        DGA corresponds to `n_delays=1`.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the backward mean first passage time at each frame of
        the trajectory.

    """
    return backward_feynman_kac(
        basis,
        weights,
        in_domain,
        1.0,
        guess,
        delay,
        n_delays,
        test_basis=test_basis,
    )


def backward_feynman_kac(
    basis, weights, in_domain, function, guess, delay, n_delays, test_basis=None
):
    """Solve the backward Feynman-Kac formula using Hankel DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the solution of the Feynman-Kac formula.
        Must be zero outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    function : list of (n_frames[i]-1,) ndarray of float
        Function to integrate. Note that is defined over transitions,
        not frames.
    guess : list of (n_frames[i],) ndarray of float
        Guess of the solution. Must obey boundary conditions.
    delay : int
        Difference in lag time (in frames) between matrix evaluations.
    n_delays : int
        Number of different lag times to use. Must be an odd integer.
        The maximum lag time is `n_delays*delay`.
        DGA corresponds to `n_delays=1`.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the solution of the backward Feynman-Kac formula at
        each frame of the trajectory.

    """
    weights = utils.shift_weights(weights, delay * n_delays)
    stat = dgastat.BackwardFeynmanKac(
        basis, weights, in_domain, function, guess, test_basis=test_basis
    )
    algo = HankelDGA(delay, n_delays).fit(stat)
    return algo.solution(stat)


class HankelDGA:
    def __init__(self, delay, n_delays):
        assert delay > 0
        assert n_delays > 0 and n_delays % 2 == 1
        self.delay = delay
        self.n_delays = n_delays
        self.coef = None

    def get_parameters(self):
        """
        Compute DGA matrices.

        Returns
        -------
        coef : (n_basis,) ndarray of float
            Projection coefficients.

        """
        return self.coef

    def set_parameters(self, coef):
        """
        Compute DGA matrices.

        Parameters
        ----------
        coef : (n_basis,) ndarray of float
            Projection coefficients.

        Returns
        -------
        self

        """
        self.coef = coef
        return self

    def fit(self, stat):
        """
        Fit statistic to data.

        Parameters
        ----------
        stat
            DGA statistic.

        Returns
        -------
        self

        """
        a, b = self.matrices(stat)
        coef = _hankel_solve(a, b)
        self.set_parameters(coef)
        return self

    def matrices(self, stat):
        """
        Compute DGA matrices.

        Parameters
        ----------
        stat
            DGA statistic.

        Returns
        -------
        a : list of (n_basis, n_basis) ndarray of float
            DGA matrices for the homogeneous term.
        b : list of (n_basis,) ndarray of float
            DGA matrices for the nonhomogeneous term.

        """
        a = []
        b = []
        for n in range(self.n_delays):
            # lag = 0 can be skipped because DGA matrices are zero
            lag = self.delay * (n + 1)
            a_n, b_n = stat.matrices(lag)
            a.append(a_n)
            b.append(b_n)
        return a, b

    def solution(self, stat):
        """
        Returns a stochastic approximation of the solution.

        Parameters
        ----------
        stat
            DGA statistic.

        Returns
        -------
        list of (n_frames[i],) ndarray of float
            Estimate of the solution.

        """
        if self.coef is None:
            raise ValueError
        n_blocks, _ = self.coef.shape
        out = []
        for n in range(n_blocks):
            lag = self.delay * n
            u = stat.transform(self.coef[n])
            u = stat.propagate(u, lag)
            out.append(u)
        out = [sum(u) for u in zip(*out, strict=True)]
        return out


def _hankel_solve(a_mats, b_mats):
    a_mats = list(a_mats)
    b_mats = list(b_mats)
    assert len(a_mats) == len(b_mats)
    n_delays = len(a_mats)
    assert n_delays % 2 == 1
    n_blocks = (n_delays + 1) // 2

    a_diff = np.full(n_delays + 1, None)
    a_diff[0] = 0
    a_diff[1:] = a_mats
    a_diff = np.diff(a_diff)

    a = np.full((n_blocks, n_blocks), None)
    for i in range(n_blocks):
        for j in range(n_blocks):
            a[i, j] = a_diff[i + j]
    a = linalg.block(a.tolist())

    b = np.full(n_delays + 1, None)
    b[0] = 0
    b[1:] = b_mats
    b = b[n_blocks:] - b[:n_blocks]
    b = np.concatenate(b.tolist())

    coef = -linalg.solve(a, b)
    return coef.reshape(n_blocks, -1)
