"""DGA with time-convolutionless memory estimators for statistics."""

from . import dgastat, linalg, utils
from .utils import normalize_weights, uniform_weights

__all__ = [
    "reweight",
    "forward_feynman_kac",
    "backward_feynman_kac",
    "DGAWithMemoryTCL",
]


def reweight(
    basis,
    lag1,
    lag2,
    *,
    maxlag=None,
    guess=None,
    test_basis=None,
    normalize=True,
):
    if maxlag is None:
        maxlag = max(lag1, lag2)
    if guess is None:
        guess = uniform_weights(basis, maxlag)
    stat = dgastat.StationaryDistribution(basis, guess, test_basis=test_basis)
    algo = DGAWithMemoryTCL(lag1, lag2).fit(stat)
    projection = algo.projection(stat)
    solution1 = algo.solution1(stat)
    solution2 = algo.solution2(stat)
    if normalize:
        projection = normalize_weights(projection)
        solution1 = normalize_weights(solution1)
        solution2 = normalize_weights(solution2)
    return projection, solution1, solution2


def forward_feynman_kac(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag1,
    lag2,
    *,
    test_basis=None,
):
    stat = dgastat.ForwardFeynmanKac(
        basis, weights, in_domain, function, guess, test_basis=test_basis
    )
    algo = DGAWithMemoryTCL(lag1, lag2).fit(stat)
    projection = algo.projection(stat)
    solution1 = algo.solution1(stat)
    solution2 = algo.solution2(stat)
    return projection, solution1, solution2


def backward_feynman_kac(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag1,
    lag2,
    *,
    test_basis=None,
):
    # shift weights because time 0 is at end of window
    weights = utils.shift_weights(weights, max(lag1, lag2))
    stat = dgastat.BackwardFeynmanKac(
        basis, weights, in_domain, function, guess, test_basis=test_basis
    )
    algo = DGAWithMemoryTCL(lag1, lag2).fit(stat)
    projection = algo.projection(stat)
    solution1 = algo.solution1(stat)
    solution2 = algo.solution2(stat)
    return projection, solution1, solution2


class DGAWithMemoryTCL:
    def __init__(self, lag1, lag2):
        self.lag1 = lag1
        self.lag2 = lag2
        self.p_coef = self.s_coef = None

    def get_parameters(self):
        """
        Get fitted parameters.

        Returns
        -------
        p_coef : (n_basis,) ndarray of float
            Projection coefficients.
        s_coef : (n_basis,) ndarray of float
            Solution coefficients.

        """
        return self.p_coef, self.s_coef

    def set_parameters(self, p_coef, s_coef):
        """
        Set fitted parameters.

        Parameters
        ----------
        p_coef : (n_basis,) ndarray of float
            Projection coefficients.
        s_coef : (n_basis,) ndarray of float
            Solution coefficients.

        Returns
        -------
        self

        """
        self.p_coef = p_coef
        self.s_coef = s_coef
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
        a, b, c0 = self.matrices(stat)
        p_coef, s_coef = solve(a, b, c0)
        self.set_parameters(p_coef, s_coef)
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
        a : pair of (n_basis, n_basis) {ndarray, sparse array} of float
            DGA matrices for the homogeneous term.
        b : pair of (n_basis,) {ndarray, sparse array} of float
            DGA matrices for the nonhomogeneous term.
        c0 : (n_basis, n_basis) {ndarray, sparse array} of float
            Matrix of inner products of basis functions.

        """
        a1, b1 = stat.matrices(self.lag1)
        a2, b2 = stat.matrices(self.lag2)
        c0 = stat.gram_matrix()
        return (a1, a2), (b1, b2), c0

    def projection(self, stat):
        """
        Returns the projected solution.

        Parameters
        ----------
        stat
            DGA statistic.

        Returns
        -------
        list of (n_frames[i],) ndarray of float
            Estimate of the projected solution.

        """
        p_coef, _ = self.get_parameters()
        if p_coef is None:
            raise ValueError
        return stat.transform(p_coef)

    def solution1(self, stat):
        """
        Returns a stochastic approximation of the solution at `lag1`.

        Parameters
        ----------
        stat
            DGA statistic.

        Returns
        -------
        list of (n_frames[i],) ndarray of float
            Estimate of the solution.

        """
        _, s_coef = self.get_parameters()
        if s_coef is None:
            raise ValueError
        return stat.propagate(stat.transform(s_coef), self.lag1)

    def solution2(self, stat):
        """
        Returns a stochastic approximation of the solution at `lag2`.

        Parameters
        ----------
        stat
            DGA statistic.

        Returns
        -------
        list of (n_frames[i],) ndarray of float
            Estimate of the solution.

        """
        _, s_coef = self.get_parameters()
        if s_coef is None:
            raise ValueError
        return stat.propagate(stat.transform(s_coef), self.lag2)


def solve(a, b, c0):
    """
    Solve DGA with time-convolutionless memory.

    Parameters
    ----------
    a : pair of (n_basis, n_basis) {ndarray, sparse array} of float
        DGA matrices for the homogeneous term.
    b : pair of (n_basis,) {ndarray, sparse array} of float
        DGA matrices for the nonhomogeneous term.
    c0 : (n_basis, n_basis) ndarray of float
        Matrix of inner products of basis functions.

    Returns
    -------
    p_coef : (n_basis,) ndarray of float
        Projection coefficients.
    s_coef : (mem, n_basis) ndarray of float
        Solution coefficients.

    """
    a1, a2 = a
    b1, b2 = b
    s_coef = -linalg.solve(a2 - a1, b2 - b1)
    p_coef = s_coef + linalg.solve(c0, a1 @ s_coef + b1)
    return p_coef, s_coef
