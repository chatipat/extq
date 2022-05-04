import numba as nb
import numpy as np


@nb.njit
def step1d(rate, t, x):
    rs, new_xs = rate(x)
    new_t = t + np.random.exponential(1.0 / np.sum(rs))
    i = _choice(rs)
    new_x = new_xs[i]
    return new_t, new_x


@nb.njit
def step2d(rate, t, x, y):
    rs, new_xs, new_ys = rate(x, y)
    new_t = t + np.random.exponential(1.0 / np.sum(rs))
    i = _choice(rs)
    new_x = new_xs[i]
    new_y = new_ys[i]
    return new_t, new_x, new_y


@nb.njit
def step3d(rate, t, x, y, z):
    rs, new_xs, new_ys, new_zs = rate(x, y, z)
    new_t = t + np.random.exponential(1.0 / np.sum(rs))
    i = _choice(rs)
    new_x = new_xs[i]
    new_y = new_ys[i]
    new_z = new_zs[i]
    return new_t, new_x, new_y, new_z


@nb.njit
def _choice(w):
    total = np.sum(w)
    p = np.random.rand()
    acc = 0.0
    for i in range(len(w) - 1):
        acc += w[i] / total
        if acc > p:
            return i
    return len(w) - 1
