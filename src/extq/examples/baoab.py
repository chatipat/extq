import numba as nb
import numpy as np


@nb.njit
def step1d(force, kT, dt, x, rx):
    bd = np.sqrt(0.5 * kT * dt)
    fx = force(x)
    new_rx = np.random.normal()
    new_x = x + dt * fx + bd * (rx + new_rx)
    return new_x, new_rx


@nb.njit
def step2d(force, kT, dt, x, y, rx, ry):
    bd = np.sqrt(0.5 * kT * dt)
    fx, fy = force(x, y)
    new_rx = np.random.normal()
    new_ry = np.random.normal()
    new_x = x + dt * fx + bd * (rx + new_rx)
    new_y = y + dt * fy + bd * (ry + new_ry)
    return new_x, new_y, new_rx, new_ry


@nb.njit
def step3d(force, kT, dt, x, y, z, rx, ry, rz):
    bd = np.sqrt(0.5 * kT * dt)
    fx, fy, fz = force(x, y, z)
    new_rx = np.random.normal()
    new_ry = np.random.normal()
    new_rz = np.random.normal()
    new_x = x + dt * fx + bd * (rx + new_rx)
    new_y = y + dt * fy + bd * (ry + new_ry)
    new_z = z + dt * fz + bd * (rz + new_rz)
    return new_x, new_y, new_z, new_rx, new_ry, new_rz
