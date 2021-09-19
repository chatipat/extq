import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


def solve(a, b):
    if scipy.sparse.issparse(a):
        return scipy.sparse.linalg.spsolve(a, b)
    else:
        return scipy.linalg.solve(a, b)


def transform(coeffs, basis, guess):
    return [y @ coeffs + g for y, g in zip(basis, guess)]


def extended_transform(coeffs, basis, guess):
    return [
        np.array([yi @ coeffs + gi for yi, gi in zip(y, g)])
        for y, g in zip(basis, guess)
    ]
