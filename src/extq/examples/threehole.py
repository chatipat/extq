import numba as nb
import numpy as np


@nb.njit
def potential(x, y):
    return (
        3.0 * np.exp(-(x ** 2) - (y - 1.0 / 3.0) ** 2)
        - 3.0 * np.exp(-(x ** 2) - (y - 5.0 / 3.0) ** 2)
        - 5.0 * np.exp(-((x - 1.0) ** 2) - y ** 2)
        - 5.0 * np.exp(-((x + 1.0) ** 2) - y ** 2)
        + 0.2 * x ** 4
        + 0.2 * (y - 1.0 / 3.0) ** 4
    )


@nb.njit
def force(x, y):
    fx = (
        6.0 * x * np.exp(-(x ** 2) - (y - 1.0 / 3.0) ** 2)
        - 6.0 * x * np.exp(-(x ** 2) - (y - 5.0 / 3.0) ** 2)
        - 10.0 * (x - 1.0) * np.exp(-((x - 1) ** 2) - y ** 2)
        - 10.0 * (x + 1.0) * np.exp(-((x + 1) ** 2) - y ** 2)
        - 0.8 * x ** 3
    )
    fy = (
        6.0 * (y - 1.0 / 3.0) * np.exp(-(x ** 2) - (y - 1.0 / 3.0) ** 2)
        - 6.0 * (y - 5.0 / 3.0) * np.exp(-(x ** 2) - (y - 5.0 / 3.0) ** 2)
        - 10.0 * y * np.exp(-((x - 1.0) ** 2) - y ** 2)
        - 10.0 * y * np.exp(-((x + 1.0) ** 2) - y ** 2)
        - 0.8 * (y - 1.0 / 3.0) ** 3
    )
    return fx, fy
