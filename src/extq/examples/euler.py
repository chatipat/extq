import numba as nb
import numpy as np


@nb.njit
def step1d(force, kT, dt, x):
    bd = np.sqrt(2.0 * kT * dt)
    fx = force(x)
    new_x = x + dt * fx + bd * np.random.normal()
    return new_x


@nb.njit
def step2d(force, kT, dt, x, y):
    bd = np.sqrt(2.0 * kT * dt)
    fx, fy = force(x, y)
    new_x = x + dt * fx + bd * np.random.normal()
    new_y = y + dt * fy + bd * np.random.normal()
    return new_x, new_y


@nb.njit
def step3d(force, kT, dt, x, y, z):
    bd = np.sqrt(2.0 * kT * dt)
    fx, fy, fz = force(x, y, z)
    new_x = x + dt * fx + bd * np.random.normal()
    new_y = y + dt * fy + bd * np.random.normal()
    new_z = z + dt * fz + bd * np.random.normal()
    return new_x, new_y, new_z
