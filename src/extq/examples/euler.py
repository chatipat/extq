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


@nb.njit
def run1d(force, kT, dt, n_steps, x, out=None):
    if out is None:
        out = np.empty((n_steps + 1, 1))
    assert out.shape == (n_steps + 1, 1)
    out[0, 0] = x
    for i in range(n_steps):
        x = step1d(force, kT, dt, x)
        out[i + 1, 0] = x
    return out


@nb.njit
def run2d(force, kT, dt, n_steps, x, y, out=None):
    if out is None:
        out = np.empty((n_steps + 1, 2))
    assert out.shape == (n_steps + 1, 2)
    out[0, 0] = x
    out[0, 1] = y
    for i in range(n_steps):
        x, y = step2d(force, kT, dt, x, y)
        out[i + 1, 0] = x
        out[i + 1, 1] = y
    return out


@nb.njit
def run3d(force, kT, dt, n_steps, x, y, z, out=None):
    if out is None:
        out = np.empty((n_steps + 1, 3))
    assert out.shape == (n_steps + 1, 3)
    out[0, 0] = x
    out[0, 1] = y
    out[0, 2] = z
    for i in range(n_steps):
        x, y, z = step3d(force, kT, dt, x, y, z)
        out[i + 1, 0] = x
        out[i + 1, 1] = y
        out[i + 1, 2] = z
    return out
