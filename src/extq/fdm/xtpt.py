import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .tpt import backward_committor
from .tpt import backward_feynman_kac
from .tpt import backward_mfpt
from .tpt import current
from .tpt import forward_committor
from .tpt import forward_feynman_kac
from .tpt import forward_mfpt
from .tpt import rate


def forward_extended_committor(
    generator,
    weights,
    transitions,
    in_domain,
    guess,
    time_transitions=None,
):
    """Compute the forward extended committor.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    weights : (n_points,) ndarray of float
        Change of measure to the invariant distribution for each point.
    transitions : (n_indices, n_indices) array-like
        Possible transitions between indices. Each element
        transitions[i, j] may be a scalar or a sparse matrix of shape
        (n_points, n_points).
    in_domain : (n_indices, n_points) ndarray of bool
        Whether each point is in the domain.
    guess : (n_indices, n_points) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    time_transitions : (n_indices, n_indices, n_points) ndarray of float
        Time-dependent transitions between indices.

    Returns
    -------
    (n_indices, n_points) ndarray of float
        Forward extended committor at each point.

    """
    gen = _extended_generator(generator, transitions, time_transitions)
    pi = np.concatenate([weights] * len(transitions))
    d = np.concatenate(in_domain)
    g = np.concatenate(guess)
    qp = forward_committor(gen, pi, d, g)
    return qp.reshape(len(transitions), len(weights))


def backward_extended_committor(
    generator,
    weights,
    transitions,
    in_domain,
    guess,
    time_transitions=None,
):
    """Compute the backward extended committor.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    weights : (n_points,) ndarray of float
        Change of measure to the invariant distribution for each point.
    transitions : (n_indices, n_indices) array-like
        Possible transitions between indices. Each element
        transitions[i, j] may be a scalar or a sparse matrix of shape
        (n_points, n_points).
    in_domain : (n_indices, n_points) ndarray of bool
        Whether each point is in the domain.
    guess : (n_indices, n_points) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    time_transitions : (n_indices, n_indices, n_points) ndarray of float
        Time-dependent transitions between indices.

    Returns
    -------
    (n_indices, n_points) ndarray of float
        Backward extended committor at each point.

    """
    gen = _extended_generator(generator, transitions, time_transitions)
    pi = np.concatenate([weights] * len(transitions))
    d = np.concatenate(in_domain)
    g = np.concatenate(guess)
    qm = backward_committor(gen, pi, d, g)
    return qm.reshape(len(transitions), len(weights))


def forward_extended_mfpt(
    generator,
    weights,
    transitions,
    in_domain,
    guess,
    time_transitions=None,
):
    """Compute the forward mean first passage time.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    weights : (n_points,) ndarray of float
        Change of measure to the invariant distribution for each point.
    transitions : (n_indices, n_indices) array-like
        Possible transitions between indices. Each element
        transitions[i, j] may be a scalar or a sparse matrix of shape
        (n_points, n_points).
    in_domain : (n_indices, n_points) ndarray of bool
        Whether each point is in the domain.
    guess : (n_indices, n_points) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.
    time_transitions : (n_indices, n_indices, n_points) ndarray of float
        Time-dependent transitions between indices.

    Returns
    -------
    (n_indices, n_points) ndarray of float
        Forward mean first passage time at each point.

    """
    gen = _extended_generator(generator, transitions, time_transitions)
    pi = np.concatenate([weights] * len(transitions))
    d = np.concatenate(in_domain)
    g = np.concatenate(guess)
    mfpt = forward_mfpt(gen, pi, d, g)
    return mfpt.reshape(len(transitions), len(weights))


def backward_extended_mfpt(
    generator,
    weights,
    transitions,
    in_domain,
    guess,
    time_transitions=None,
):
    """Compute the backward mean first passage time.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    weights : (n_points,) ndarray of float
        Change of measure to the invariant distribution for each point.
    transitions : (n_indices, n_indices) array-like
        Possible transitions between indices. Each element
        transitions[i, j] may be a scalar or a sparse matrix of shape
        (n_points, n_points).
    in_domain : (n_indices, n_points) ndarray of bool
        Whether each point is in the domain.
    guess : (n_indices, n_points) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.
    time_transitions : (n_indices, n_indices, n_points) ndarray of float
        Time-dependent transitions between indices.

    Returns
    -------
    (n_indices, n_points) ndarray of float
        Backward mean first passage time at each point.

    """
    gen = _extended_generator(generator, transitions, time_transitions)
    pi = np.concatenate([weights] * len(transitions))
    d = np.concatenate(in_domain)
    g = np.concatenate(guess)
    mfpt = backward_mfpt(gen, pi, d, g)
    return mfpt.reshape(len(transitions), len(weights))


def forward_extended_feynman_kac(
    generator,
    weights,
    transitions,
    in_domain,
    function,
    guess,
    time_transitions=None,
):
    """Solve the forward Feynman-Kac formula.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    weights : (n_points,) ndarray of float
        Change of measure to the invariant distribution for each point.
    transitions : (n_indices, n_indices) array-like
        Possible transitions between indices. Each element
        transitions[i, j] may be a scalar or a sparse matrix of shape
        (n_points, n_points).
    in_domain : (n_indices, n_points) ndarray of bool
        Whether each point is in the domain.
    function : (n_indices, n_points) ndarray of float
        Function to integrate. Must be zero outside of the domain.
    guess : (n_indices, n_points) ndarray of float
        Guess for the solution. Must obey boundary conditions.
    time_transitions : (n_indices, n_indices, n_points) ndarray of float
        Time-dependent transitions between indices.

    Returns
    -------
    (n_indices, n_points) ndarray of float
        Solution of the Feynman-Kac formula at each point.

    """
    gen = _extended_generator(generator, transitions, time_transitions)
    pi = np.concatenate([weights] * len(transitions))
    d = np.concatenate(in_domain)
    f = np.concatenate(function)
    g = np.concatenate(guess)
    soln = forward_feynman_kac(gen, pi, d, f, g)
    return soln.reshape(len(transitions), len(weights))


def backward_extended_feynman_kac(
    generator,
    weights,
    transitions,
    in_domain,
    function,
    guess,
    time_transitions=None,
):
    """Solve the backward Feynman-Kac formula.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    weights : (n_points,) ndarray of float
        Change of measure to the invariant distribution for each point.
    transitions : (n_indices, n_indices) array-like
        Possible transitions between indices. Each element
        transitions[i, j] may be a scalar or a sparse matrix of shape
        (n_points, n_points).
    in_domain : (n_indices, n_points) ndarray of bool
        Whether each point is in the domain.
    function : (n_indices, n_points) ndarray of float
        Function to integrate. Must be zero outside of the domain.
    guess : (n_indices, n_points) ndarray of float
        Guess for the solution. Must obey boundary conditions.
    time_transitions : (n_indices, n_indices, n_points) ndarray of float
        Time-dependent transitions between indices.

    Returns
    -------
    (n_indices, n_points) ndarray of float
        Solution of the Feynman-Kac formula at each point.

    """
    gen = _extended_generator(generator, transitions, time_transitions)
    pi = np.concatenate([weights] * len(transitions))
    d = np.concatenate(in_domain)
    f = np.concatenate(function)
    g = np.concatenate(guess)
    soln = backward_feynman_kac(gen, pi, d, f, g)
    return soln.reshape(len(transitions), len(weights))


def extended_rate(
    generator,
    forward_q,
    backward_q,
    weights,
    transitions,
    rxn_coords=None,
    time_transitions=None,
):
    """Compute the TPT rate using extended committors.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    forward_q : (n_indices, n_points) ndarray of float
        Forward extended committor at each point.
    backward_q : (n_indices, n_points) ndarray of float
        Backward extended committor at each point.
    weights : (n_points,) ndarray of float.
        Change of measure to the invariant distribution at each point.
    rxn_coords : (n_indices, n_points) ndarray of float, optional
        Reaction coordinate at each point. This must be zero in the
        reactant state and one in the product state. If None, estimate
        the rate without using a reaction coordinate.
    time_transitions : (n_indices, n_indices, n_points) ndarray of float
        Time-dependent transitions between indices.

    Returns
    -------
    float
        TPT rate.

    """
    gen = _extended_generator(generator, transitions, time_transitions)
    qp = np.concatenate(forward_q)
    qm = np.concatenate(backward_q)
    pi = np.concatenate([weights] * len(transitions))
    if rxn_coords is None:
        h = None
    else:
        h = np.concatenate(rxn_coords)
    r = rate(gen, qp, qm, pi, h)
    return r * len(transitions)


def extended_current(
    generator,
    forward_q,
    backward_q,
    weights,
    transitions,
    cv,
    time_transitions=None,
):
    """Compute the reactive current using extended committors.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    forward_q : (n_indices, n_points) ndarray of float
        Forward extended committor at each point.
    backward_q : (n_indices, n_points) ndarray of float
        Backward extended committor at each point.
    weights : (n_points,) ndarray of float.
        Change of measure to the invariant distribution at each point.
    transitions : (n_indices, n_indices) array-like
        Possible transitions between indices. Each element
        transitions[i, j] may be a scalar or a sparse matrix of shape
        (n_points, n_points).
    rxn_coords : (n_indices, n_points) ndarray of float
        Collective variable at each point.
    time_transitions : (n_indices, n_indices, n_points) ndarray of float
        Time-dependent transitions between indices.

    Returns
    -------
    (n_indices, n_points) ndarray of float
        Reactive current at each point.

    """
    gen = _extended_generator(generator, transitions, time_transitions)
    qp = np.concatenate(forward_q)
    qm = np.concatenate(backward_q)
    pi = np.concatenate([weights] * len(transitions))
    h = np.concatenate(cv)
    j = current(gen, qp, qm, pi, h)
    return j.reshape(len(transitions), len(weights)) * len(transitions)


def _extended_generator(generator, transitions, time_transitions=None):
    """Compute the generator for extended DGA/TPT.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    transitions : (n_indices, n_indices) array-like
        Possible transitions between indices. Each element
        transitions[i, j] may be a scalar or a sparse matrix of shape
        (n_points, n_points).
    time_transitions : (n_indices, n_indices, n_points) ndarray of float
        Time-dependent transitions between indices.

    """
    # time-independent term
    xgen = scipy.sparse.bmat(
        [[generator.multiply(mij) for mij in mi] for mi in transitions],
        format="csr",
    )
    # time-dependent term
    if time_transitions is not None:
        xgen += scipy.sparse.bmat(
            [
                [scipy.sparse.diags(mij) for mij in mi]
                for mi in time_transitions
            ],
            format="csr",
        )
    return xgen
