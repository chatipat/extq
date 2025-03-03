import numpy as np

from . import linalg


def vac(basis, weights, lag, *, num_evecs=None, force_dense=True):
    assert lag > 0
    n_basis = None
    c1 = 0.0
    c0 = 0.0
    for x, w in zip(basis, weights, strict=True):
        n_frames = x.shape[0]
        n_basis = x.shape[1] if n_basis is None else n_basis
        assert x.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)

        # note that if lag >= n_frames, all weights must be zero
        if not np.any(w):  # np.all(w == 0)
            continue
        assert not np.any(w[-lag:])  # np.all(w[-lag:] == 0)
        # after this, at least one frame has nonzero weight

        # more efficient implementation of
        #   c1 += x[:-lag].T @ linalg.scale_rows(w[:-lag], x[lag:])
        #   c1 += x[lag:].T @ linalg.scale_rows(w[:-lag], x[:-lag])
        #   c0 += x[:-lag].T @ linalg.scale_rows(w[:-lag], x[:-lag])
        #   c0 += x[lag:].T @ linalg.scale_rows(w[:-lag], x[lag:])
        c1_traj = x[:-lag].T @ linalg.scale_rows(w[:-lag], x[lag:])
        c1_traj = c1_traj + c1_traj.T
        c0_traj = x.T @ linalg.scale_rows(w + np.roll(w, lag), x)
        if force_dense:
            c1_traj = linalg.as_dense(c1_traj)
            c0_traj = linalg.as_dense(c0_traj)
        c1 += c1_traj
        c0 += c0_traj
    eigvals, coefs = linalg.eigh(c1, c0, k=num_evecs)
    eigvecs = [x @ coefs for x in basis]
    return eigvals, eigvecs


def implied_timescales(eigvals, lag):
    """
    Calculate implied timescales from VAC eigenvalues.

    Parameters
    ----------
    eigvals : ndarray of float
        VAC eigenvalues.
    lag : {int, float, ndarray of {int, float}}
        Lag time.

    Returns
    -------
    ndarray of float
        Implied timescales.

    """
    return -lag / np.log(eigvals)
