import numba as nb


@nb.njit
def potential(x):
    return (x**2 - 1.0) ** 2 * (10.0 * x**2 + 1.0)


@nb.njit
def force(x):
    return -60.0 * x**5 + 76.0 * x**3 - 16.0 * x
