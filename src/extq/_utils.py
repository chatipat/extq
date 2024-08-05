import numpy as np


def sum_windows(a, start, end):
    r"""
    Return the sum of the elements within each window.

    The output of this function is::

        out[i] = np.sum(a[start[i] : end[i]], axis=0)

    Parameters
    ----------
    a : (n, *shape) array_like
        Input array.
    start : (n_windows,) ndarray of int
        Starting index (inclusive) of each window.
    end : (n_windows,) ndarray of int
        Ending index (exclusive) of each window.

    Returns
    -------
    out : (n_windows, *shape) ndarray
        Sum of the elements within each window.

    """
    c = np.concatenate([
        np.zeros((1, *a.shape[1:]), a.dtype),
        np.cumsum(a, axis=0),
    ])
    return c[end] - c[start]


def count_transition_paths(in_domain, reactant, product):
    r"""
    Count the number of complete transition paths within the trajectory.

    Parameters
    ----------
    in_domain : (n,) ndarray of bool
        Whether each frame is in the domain.
    reactant : (n,) ndarray of {bool, int, float}
        Whether each frame is in the reactant.
    product : (n,) ndarray of {bool, int, float}
        Whether each frame is in the product.

    Returns
    -------
    int or float
        Number of complete transition paths.

    """
    assert in_domain.shape == reactant.shape == product.shape
    (t,) = np.nonzero(in_domain)
    return np.sum(reactant[t[:-1]] * product[t[1:]])


def count_transition_paths_windows(in_domain, reactant, product, start, end):
    r"""
    Count the number of complete transition paths within each window.

    Parameters
    ----------
    in_domain : (n,) ndarray of bool
        Whether each frame is in the domain.
    reactant : (n,) ndarray of {bool, int, float}
        Whether each frame is in the reactant.
    product : (n,) ndarray of {bool, int, float}
        Whether each frame is in the product.
    start : (n_windows,) ndarray of int
        Starting index (inclusive) of each window.
    end : (n_windows,) ndarray of int
        Ending index (exclusive) of each window.

    Returns
    -------
    ndarray of {int, float}
        Number of complete transition paths within each window.

    """
    assert in_domain.shape == reactant.shape == product.shape
    assert start.shape == end.shape
    (t,) = np.nonzero(in_domain)

    # whether each segment t[i],...,t[i+1] is a transition path
    is_transition_path = reactant[t[:-1]] * product[t[1:]]

    # number of initial times of transition paths before edge
    count_initial = np.zeros(len(in_domain), dtype=is_transition_path.dtype)
    count_initial[t[:-1]] = is_transition_path
    count_initial = np.concatenate([[0], np.cumsum(count_initial)])

    # number of final times of transition paths before edge
    count_final = np.zeros(len(in_domain), dtype=is_transition_path.dtype)
    count_final[t[1:]] = is_transition_path
    count_final = np.concatenate([[0], np.cumsum(count_final)])

    out = count_final[end] - count_initial[start]

    # when window k is wholly within transition path i,
    # out[k] is transition_paths[i] less than the correct answer
    # because of double subtraction
    t = np.concatenate([[-1], t, [len(in_domain)]])
    is_transition_path = np.concatenate([[0], is_transition_path, [0]])
    # idx = segment index of each edge
    idx = np.repeat(np.arange(len(is_transition_path)), np.diff(t))
    idx_start = idx[start]
    idx_end = idx[end]
    mask = idx_start == idx_end  # window is wholly within transition path
    out[mask] += is_transition_path[idx_start[mask]]

    return out
