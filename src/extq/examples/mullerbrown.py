import numba as nb
import numpy as np


@nb.njit
def potential(x, y):
    u1 = _potential_term(-200.0, -1.0, 0.0, -10.0, 1.0, 0.0, x, y)
    u2 = _potential_term(-100.0, -1.0, 0.0, -10.0, 0.0, 0.5, x, y)
    u3 = _potential_term(-170.0, -6.5, 11.0, -6.5, -0.5, 1.5, x, y)
    u4 = _potential_term(15.0, 0.7, 0.6, 0.7, -1.0, 1.0, x, y)
    return u1 + u2 + u3 + u4


@nb.njit
def force(x, y):
    fx1, fy1 = _force_term(-200.0, -1.0, 0.0, -10.0, 1.0, 0.0, x, y)
    fx2, fy2 = _force_term(-100.0, -1.0, 0.0, -10.0, 0.0, 0.5, x, y)
    fx3, fy3 = _force_term(-170.0, -6.5, 11.0, -6.5, -0.5, 1.5, x, y)
    fx4, fy4 = _force_term(15.0, 0.7, 0.6, 0.7, -1.0, 1.0, x, y)
    return fx1 + fx2 + fx3 + fx4, fy1 + fy2 + fy3 + fy4


@nb.njit
def _potential_term(A, a, b, c, x0, y0, x, y):
    dx = x - x0
    dy = y - y0
    return A * np.exp(a * dx ** 2 + b * dx * dy + c * dy ** 2)


@nb.njit
def _force_term(A, a, b, c, x0, y0, x, y):
    dx = x - x0
    dy = y - y0
    u = A * np.exp(a * dx ** 2 + b * dx * dy + c * dy ** 2)
    fx = -u * (2.0 * a * dx + b * dy)
    fy = -u * (b * dx + 2.0 * c * dy)
    return fx, fy
