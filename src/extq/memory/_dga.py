import numpy as np

from .. import linalg
from . import _matrix
from . import _memory


def reweight(basis, weights, lag, mem=0, test=None):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(_matrix.reweight_matrix(basis, weights, t, test=test))
    return _reweight(mats, basis, weights)


def _reweight(mats, basis, weights):
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
    coeffs = reweight_solve(gen)
    return reweight_transform(coeffs, basis, weights)


def forward_committor(basis, weights, in_domain, guess, lag, mem=0, test=None):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_committor_matrix(
                basis, weights, in_domain, guess, t, test=test
            )
        )
    return _forward(mats, basis, in_domain, guess)


def forward_mfpt(basis, weights, in_domain, guess, lag, mem=0, test=None):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_mfpt_matrix(
                basis, weights, in_domain, guess, t, test=test
            )
        )
    return _forward(mats, basis, in_domain, guess)


def forward_feynman_kac(
    basis, weights, in_domain, function, guess, lag, mem=0, test=None
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_feynman_kac_matrix(
                basis, weights, in_domain, function, guess, t, test=test
            )
        )
    return _forward(mats, basis, in_domain, guess)


def _forward(mats, basis, in_domain, guess):
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
    coeffs = forward_solve(gen)
    return forward_transform(coeffs, basis, in_domain, guess)


def backward_committor(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.backward_committor_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _backward(mats, w_basis, basis, in_domain, guess)


def backward_mfpt(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.backward_mfpt_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _backward(mats, w_basis, basis, in_domain, guess)


def backward_feynman_kac(
    w_basis,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.backward_feynman_kac_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                function,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _backward(mats, w_basis, basis, in_domain, guess)


def _backward(mats, w_basis, basis, in_domain, guess):
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
    coeffs = backward_solve(gen)
    return backward_transform(coeffs, w_basis, basis, in_domain, guess)


def reweight_integral(basis, weights, values, lag, mem=0, test=None):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.reweight_integral_matrix(
                basis, weights, values, t, test=test
            )
        )
    return _integral(mats, lag, mem)


def forward_committor_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_committor_integral_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                values,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _integral(mats, lag, mem)


def forward_mfpt_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_mfpt_integral_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                values,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _integral(mats, lag, mem)


def forward_feynman_kac_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_feynman_kac_integral_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                values,
                function,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _integral(mats, lag, mem)


def backward_committor_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        _matrix.backward_committor_integral_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            values,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
    return _integral(mats, lag, mem)


def backward_mfpt_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        _matrix.backward_mfpt_integral_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            values,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
    return _integral(mats, lag, mem)


def backward_feynman_kac_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        _matrix.backward_feynman_kac_integral_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            values,
            function,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
    return _integral(mats, lag, mem)


def tpt_integral(
    w_basis,
    b_basis,
    f_basis,
    weights,
    in_domain,
    values,
    b_guess,
    f_guess,
    lag,
    mem=0,
    w_test=None,
    b_test=None,
    f_test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.tpt_integral_matrix(
                w_basis,
                b_basis,
                f_basis,
                weights,
                in_domain,
                values,
                b_guess,
                f_guess,
                t,
                w_test=w_test,
                b_test=b_test,
                f_test=f_test,
            )
        )
    return _integral(mats, lag, mem)


def integral(
    w_basis,
    b_basis,
    f_basis,
    weights,
    b_domain,
    f_domain,
    values,
    b_function,
    f_function,
    b_guess,
    f_guess,
    lag,
    mem=0,
    w_test=None,
    b_test=None,
    f_test=None,
):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.integral_matrix(
                w_basis,
                b_basis,
                f_basis,
                weights,
                b_domain,
                f_domain,
                values,
                b_function,
                f_function,
                b_guess,
                f_guess,
                t,
                w_test=w_test,
                b_test=b_test,
                f_test=f_test,
            )
        )
    return _integral(mats, lag, mem)


def _integral(mats, lag, mem):
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
    eye = _memory.identity(mats, mems)
    return integral_solve(gen, eye) / (lag // (mem + 1))


def reweight_solve(gen):
    """Solve problem for change of measure.

    Parameters
    ----------
    gen : ndarray of float
        Magic generator operator + memory

    Returns
    -------
    ndarray
        Coefficients
    """
    return np.concatenate([[1.0], linalg.solve(gen.T[1:, 1:], -gen.T[1:, 0])])


def reweight_transform(coeffs, basis, weights):
    result = []
    for x_w, w in zip(basis, weights):
        result.append(w * (coeffs[0] + x_w @ coeffs[1:]))
    return result


def forward_solve(gen):
    """Solve problem for forward-in-time statistic.

    Parameters
    ----------
    gen : ndarray of float
        Magic generator operator + memory

    Returns
    -------
    ndarray
        Coefficients
    """
    return np.concatenate([linalg.solve(gen[:-1, :-1], -gen[:-1, -1]), [1.0]])


def forward_transform(coeffs, basis, in_domain, guess):
    result = []
    for y_f, d_f, g_f in zip(basis, in_domain, guess):
        result.append(g_f + np.where(d_f, y_f @ coeffs[:-1], 0.0) / coeffs[-1])
    return result


def backward_solve(gen):
    """Solve problem for backward-in-time statistic.

    Parameters
    ----------
    gen : ndarray of float
        Magic generator operator + memory

    Returns
    -------
    ndarray
        Coefficients
    """
    return np.concatenate([[1.0], linalg.solve(gen.T[1:, 1:], -gen.T[1:, 0])])


def backward_transform(coeffs, w_basis, basis, in_domain, guess):
    result = []
    for x_w, x_b, d_b, g_b in zip(w_basis, basis, in_domain, guess):
        n = x_w.shape[1] + 1
        com = coeffs[0] + x_w @ coeffs[1:n]
        result.append(g_b + np.where(d_b, x_b @ coeffs[n:], 0.0) / com)
    return result


def integral_solve(gen, eye):
    """Solve problem for integral average statistic.

    Parameters
    ----------
    gen : ndarray of float
        Magic generator operator + memory
    eye : ndarray of float

    Returns
    -------
    ndarray
        Integral result

    """
    mat = linalg.solve(eye, gen)
    forward_coeffs = np.concatenate(
        [linalg.solve(mat[1:-1, 1:-1], -mat[1:-1, -1]), [1.0]]
    )
    backward_coeffs = np.concatenate(
        [[1.0], linalg.solve(mat.T[1:-1, 1:-1], -mat.T[1:-1, 0])]
    )
    return backward_coeffs @ mat[:-1, 1:] @ forward_coeffs


def _memlags(lag, mem):
    """Returns the lag times at which to evaluate correlation matrices.

    This function acts similarly to `numpy.linspace(0, lag, mem+2)`.

    Parameters
    ----------
    lag : int
        Maximum lag time.
    mem : int
        Number of memory matrices, which are evaluated at equally spaced
        times between time 0 and time `lag`. `mem+1` must evenly divide
        `lag`. For example, with a `lag=32`, `mem=3` and `mem=7` are
        fine since 7+1=8 and 3+1=4 evenly divide 32.

    Returns
    -------
    iterable of int
        Lag times at which to evaluate correlation matrices.

    """
    assert lag % (mem + 1) == 0
    return np.arange(0, lag + 1, lag // (mem + 1))
