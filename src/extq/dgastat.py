import numpy as np

from . import _dgamat, _utils


class StationaryDistribution:
    def __init__(self, basis, weights, test_basis=None):
        self.basis = basis
        self.weights = weights
        self.test_basis = test_basis

    def matrices(self, lag):
        return _dgamat.reweight_matrices(
            self.basis, lag, self.weights, test_basis=self.test_basis
        )

    def gram_matrix(self):
        return _dgamat.gram_matrix(self.basis, self.weights, test_basis=self.test_basis)

    def transform(self, coef):
        return [
            w * (y @ coef + 1.0) for y, w in zip(self.basis, self.weights, strict=True)
        ]

    def transform_difference(self, coef):
        return [w * (y @ coef) for y, w in zip(self.basis, self.weights, strict=True)]

    def propagate(self, guess, lag):
        return [_utils.distribution_propagate(u, lag) for u in guess]


class ForwardFeynmanKac:
    def __init__(self, basis, weights, in_domain, function, guess, test_basis=None):
        function = _broadcast_integrand(function, guess)
        self.basis = basis
        self.weights = weights
        self.in_domain = in_domain
        self.function = function
        self.guess = guess
        self.test_basis = test_basis

    def matrices(self, lag):
        return _dgamat.forward_feynman_kac_matrices(
            self.basis,
            self.weights,
            self.in_domain,
            self.function,
            self.guess,
            lag,
            test_basis=self.test_basis,
        )

    def gram_matrix(self):
        return _dgamat.gram_matrix(self.basis, self.weights, test_basis=self.test_basis)

    def transform(self, coef):
        return [y @ coef + g for y, g in zip(self.basis, self.guess, strict=True)]

    def transform_difference(self, coef):
        return [y @ coef for y in self.basis]

    def propagate(self, guess, lag):
        return [
            _utils.forward_feynman_kac_propagate(u, d, f, lag)
            for u, d, f in zip(guess, self.in_domain, self.function, strict=True)
        ]


class BackwardFeynmanKac:
    def __init__(self, basis, weights, in_domain, function, guess, test_basis=None):
        function = _broadcast_integrand(function, guess)
        self.basis = basis
        self.weights = weights
        self.in_domain = in_domain
        self.function = function
        self.guess = guess
        self.test_basis = test_basis

    def matrices(self, lag):
        return _dgamat.backward_feynman_kac_matrices(
            self.basis,
            self.weights,
            self.in_domain,
            self.function,
            self.guess,
            lag,
            test_basis=self.test_basis,
        )

    def gram_matrix(self):
        return _dgamat.gram_matrix(self.basis, self.weights, test_basis=self.test_basis)

    def transform(self, coef):
        return [y @ coef + g for y, g in zip(self.basis, self.guess, strict=True)]

    def transform_difference(self, coef):
        return [y @ coef for y in self.basis]

    def propagate(self, guess, lag):
        return [
            _utils.backward_feynman_kac_propagate(u, d, f, lag)
            for u, d, f in zip(guess, self.in_domain, self.function, strict=True)
        ]


def _broadcast_integrand(f, trajs):
    if not np.iterable(f):
        f = [np.broadcast_to(f, traj.shape[0] - 1) for traj in trajs]
    return f
