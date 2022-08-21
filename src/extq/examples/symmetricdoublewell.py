import numba as nb


@nb.njit
def potential(x):
    return (x**2 - 1.0) ** 2


@nb.njit
def force(x):
    return -4.0 * x * (x**2 - 1.0)
