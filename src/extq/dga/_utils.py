import numpy as np


def transform(coeffs, basis, guess):
    return [y @ coeffs + g for y, g in zip(basis, guess)]


def extended_transform(coeffs, basis, guess):
    return [
        np.array([yi @ coeffs + gi for yi, gi in zip(y, g)])
        for y, g in zip(basis, guess)
    ]
