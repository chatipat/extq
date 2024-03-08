import numba as nb
import numpy as np


@nb.njit
def potential(x, y, z):
    box = ((x - 1) / 2) ** 4 + (y / 3) ** 4 + (z / 3) ** 4
    a = -3 * np.exp(-(x**2 + (y - 2) ** 2 + (z - 2) ** 2))
    b = -3 * np.exp(-(x**2 + (y + 2) ** 2 + (z + 2) ** 2))
    c = -np.exp(-((x - 2) ** 2 + y**2))
    s1 = -2 * np.exp(-(x**2 + y**2 + (z - 2) ** 2))
    s2 = -2 * np.exp(-(x**2 + y**2 + (z + 2) ** 2))
    u = 5 * (box + a + b + c + s1 + s2)
    return u


@nb.njit
def force(x, y, z):
    box_dx = (4 / 2**4) * (x - 1) ** 3
    box_dy = (4 / 3**4) * y**3
    box_dz = (4 / 3**4) * z**3

    a = -3 * np.exp(-(x**2 + (y - 2) ** 2 + (z - 2) ** 2))
    a_dx = -2 * x * a
    a_dy = -2 * (y - 2) * a
    a_dz = -2 * (z - 2) * a

    b = -3 * np.exp(-(x**2 + (y + 2) ** 2 + (z + 2) ** 2))
    b_dx = -2 * x * b
    b_dy = -2 * (y + 2) * b
    b_dz = -2 * (z + 2) * b

    c = -np.exp(-((x - 2) ** 2 + y**2))
    c_dx = -2 * (x - 2) * c
    c_dy = -2 * y * c

    s1 = -2 * np.exp(-(x**2 + y**2 + (z - 2) ** 2))
    s1_dx = -2 * x * s1
    s1_dy = -2 * y * s1
    s1_dz = -2 * (z - 2) * s1

    s2 = -2 * np.exp(-(x**2 + y**2 + (z + 2) ** 2))
    s2_dx = -2 * x * s2
    s2_dy = -2 * y * s2
    s2_dz = -2 * (z + 2) * s2

    fx = -5 * (box_dx + a_dx + b_dx + c_dx + s1_dx + s2_dx)
    fy = -5 * (box_dy + a_dy + b_dy + c_dy + s1_dy + s2_dy)
    fz = -5 * (box_dz + a_dz + b_dz + s1_dz + s2_dz)
    return fx, fy, fz
