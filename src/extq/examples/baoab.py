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


@nb.njit
def run1d(force, kT, dt, n_steps, x, out=None):
    if out is None:
        out = np.empty((n_steps + 1, 1))
    assert out.shape == (n_steps + 1, 1)
    rx = np.random.normal()
    out[0, 0] = x
    for i in range(n_steps):
        x, rx = step1d(force, kT, dt, x, rx)
        out[i + 1, 0] = x
    return out


@nb.njit
def run2d(force, kT, dt, n_steps, x, y, out=None):
    if out is None:
        out = np.empty((n_steps + 1, 2))
    assert out.shape == (n_steps + 1, 2)
    rx = np.random.normal()
    ry = np.random.normal()
    out[0, 0] = x
    out[0, 1] = y
    for i in range(n_steps):
        x, y, rx, ry = step2d(force, kT, dt, x, y, rx, ry)
        out[i + 1, 0] = x
        out[i + 1, 1] = y
    return out


@nb.njit
def run3d(force, kT, dt, n_steps, x, y, z, out=None):
    if out is None:
        out = np.empty((n_steps + 1, 3))
    assert out.shape == (n_steps + 1, 3)
    rx = np.random.normal()
    ry = np.random.normal()
    rz = np.random.normal()
    out[0, 0] = x
    out[0, 1] = y
    out[0, 2] = z
    for i in range(n_steps):
        x, y, z, rx, ry, rz = step3d(force, kT, dt, x, y, z, rx, ry, rz)
        out[i + 1, 0] = x
        out[i + 1, 1] = y
        out[i + 1, 2] = z
    return out
