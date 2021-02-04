import numba as nb
import numpy as np


@nb.njit
def potential(x, y):
    return 2.5 * (x ** 2 - 1.0) ** 2 + 5.0 * y ** 2


@nb.njit
def force(x, y):
    fx = -10.0 * x * (x ** 2 - 1.0)
    fy = -10.0 * y
    return fx, fy
