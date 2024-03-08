import numba as nb
import numpy as np


@nb.njit
def potential(x, y, z):
    u = 5 * (
        (x / 2) ** 4
        + (y / 3) ** 4
        + (z / 2) ** 4
        - 3 * np.exp(-(x**2 + (y - 3) ** 2))
        - 3 * np.exp(-(x**2 + (y + 3) ** 2))
        - 2 * np.exp(-((x + y) ** 2 + (z + 1) ** 2))
        - 2 * np.exp(-((x - y) ** 2 + (z - 1) ** 2))
        + np.exp(-(y**2))
    )
    return u


@nb.njit
def force(x, y, z):
    box_x = (4 / 2**4) * x**3
    box_y = (4 / 3**4) * y**3
    box_z = (4 / 2**4) * z**3

    a = -3 * np.exp(-(x**2 + (y - 3) ** 2))
    a_x = -2 * x * a
    a_y = -2 * (y - 3) * a

    b = -3 * np.exp(-(x**2 + (y + 3) ** 2))
    b_x = -2 * x * b
    b_y = -2 * (y + 3) * b

    c = -2 * np.exp(-((x + y) ** 2 + (z + 1) ** 2))
    c_x = -2 * (x + y) * c
    c_y = -2 * (y + x) * c
    c_z = -2 * (z + 1) * c

    d = -2 * np.exp(-((x - y) ** 2 + (z - 1) ** 2))
    d_x = -2 * (x - y) * d
    d_y = -2 * (y - x) * d
    d_z = -2 * (z - 1) * d

    e = np.exp(-(y**2))
    e_y = -2 * y * e

    fx = -5 * (box_x + a_x + b_x + c_x + d_x)
    fy = -5 * (box_y + a_y + b_y + c_y + d_y + e_y)
    fz = -5 * (box_z + c_z + d_z)

    return fx, fy, fz
