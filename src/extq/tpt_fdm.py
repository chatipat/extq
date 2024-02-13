import numpy as np

__all__ = [
    "rate",
    "current",
    "integral",
    "pointwise_integral",
]


def rate(
    generator, forward_q, backward_q, weights, rxn_coords=None, normalize=True
):
    """Compute the TPT rate.

    Parameters
    ----------
    generator : (M, M) sparse matrix of float
        Generator matrix.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float
        Change of measure at each point.
    rxn_coords : (M,) ndarray of float, optional
        Reaction coordinate at each point. This must be zero in the
        reactant state and one in the product state. If None, estimate
        the rate without using a reaction coordinate.
    normalize : bool, optional
        If True (default), normalize `weights` to one.

    Returns
    -------
    float
        TPT rate.

    """
    w, qp, qm = np.broadcast_arrays(weights, forward_q, backward_q)

    size = w.size
    shape = w.shape
    assert generator.shape == (size, size)

    pi_qm = np.ravel(w * qm)
    qp = np.ravel(qp)

    if rxn_coords is None:
        out = pi_qm @ generator @ qp
    else:
        h = np.broadcast_to(rxn_coords, shape)
        h = np.ravel(h)
        out = pi_qm @ (generator @ (qp * h) - h * (generator @ qp))
    if normalize:
        out /= np.sum(weights)
    return out


def current(generator, forward_q, backward_q, weights, cv, normalize=True):
    """Compute the reactive current at each point.

    Parameters
    ----------
    generator : (M, M) sparse matrix of float
        Generator matrix.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float.
        Change of measure at each point.
    cv : (M,) ndarray of float
        Collective variable at each point.
    normalize : bool
        If True (default), normalize `weights` to one.

    Returns
    -------
    (M,) ndarray of float
        Reactive current at each point.

    """
    w, qp, qm, h = np.broadcast_arrays(weights, forward_q, backward_q, cv)

    size = w.size
    shape = w.shape
    assert generator.shape == (size, size)

    pi_qm = np.ravel(w * qm)
    qp = np.ravel(qp)
    h = np.ravel(h)

    forward_flux = pi_qm * (generator @ (qp * h) - h * (generator @ qp))
    backward_flux = ((pi_qm * h) @ generator - (pi_qm @ generator) * h) * qp
    out = 0.5 * (forward_flux - backward_flux)
    if normalize:
        out /= np.sum(weights)
    return out.reshape(shape)


def integral(generator, forward_q, backward_q, weights, normalize=True):
    """Integrate a TPT objective function over the reaction ensemble.

    Parameter
    ---------
    generator : (M, M) sparse matrix of float
        Generator matrix, modified to encode the TPT objective function.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float.
        Change of measure at each point.
    normalize : bool
        If True (default), normalize `weights` to one.

    Returns
    -------
    float
        Integral of the objective function over the reaction ensemble.

    """
    w, qp, qm = np.broadcast_arrays(weights, forward_q, backward_q)

    size = w.size
    assert generator.shape == (size, size)

    pi_qm = np.ravel(w * qm)
    qp = np.ravel(qp)

    out = pi_qm @ generator @ qp
    if normalize:
        out /= np.sum(weights)
    return out


def pointwise_integral(
    generator, forward_q, backward_q, weights, normalize=True
):
    """Calculate the contribution of each point to a TPT integral.

    Parameter
    ---------
    generator : (M, M) sparse matrix of float
        Generator matrix, modified to encode the TPT objective function.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float.
        Change of measure at each point.
    normalize : bool, optional
        If True (default), normalize `weights` to one.

    Returns
    -------
    (M,) ndarray of float
        Contribution of each point to the TPT integral.

    """
    w, qp, qm = np.broadcast_arrays(weights, forward_q, backward_q)

    size = w.size
    shape = w.shape
    assert generator.shape == (size, size)

    pi_qm = np.ravel(w * qm)
    qp = np.ravel(qp)

    out = 0.5 * (pi_qm * (generator @ qp) + (pi_qm @ generator) * qp)
    if normalize:
        out /= np.sum(weights)
    return out.reshape(shape)
