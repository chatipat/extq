from .dga_methods import DGA
from .dgastat import BackwardVectorFeynmanKac, ForwardVectorFeynmanKac
from .utils import shift_weights

__all__ = [
    "forward_extended_committor",
    "forward_extended_mfpt",
    "forward_extended_feynman_kac",
    "backward_extended_committor",
    "backward_extended_mfpt",
    "backward_extended_feynman_kac",
]


def forward_extended_committor(
    basis,
    weights,
    transitions,
    in_domain,
    guess,
    lag,
    test_basis=None,
    *,
    method=None,
    output="projection",
):
    """Estimate the forward extended committor using DGA.

    Parameters
    ----------
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.
    method : DGAMethod, optional
        Method for estimating the solution. If None (default), use
        ``DGA(lag)``.
    output : str, optional
        Type of output to return. The default ('projection') returns
        the projected solution.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimated forward extended committor at each frame.

    """
    return forward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        0.0,
        guess,
        lag,
        test_basis=test_basis,
        method=method,
        output=output,
    )


def forward_extended_mfpt(
    basis,
    weights,
    transitions,
    in_domain,
    guess,
    lag,
    test_basis=None,
    *,
    method=None,
    output="projection",
):
    """Estimate the forward mean first passage time using DGA.

    Parameters
    ----------
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the mean first passage time . Must be zero
        outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the mean first passage time . Must obey boundary
        conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.
    method : DGAMethod, optional
        Method for estimating the solution. If None (default), use
        ``DGA(lag)``.
    output : str, optional
        Type of output to return. The default ('projection') returns
        the projected solution.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimated forward mean first passage time at each frame.

    """
    return forward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        1.0,
        guess,
        lag,
        test_basis=test_basis,
        method=method,
        output=output,
    )


def forward_extended_feynman_kac(
    basis,
    weights,
    transitions,
    in_domain,
    function,
    guess,
    lag,
    test_basis=None,
    *,
    method=None,
    output="projection",
):
    """Solve the forward Feynman-Kac formula using DGA.

    Parameters
    ----------
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the solution to the Feynman-Kac formula.
        Must be zero outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    function : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Function to integrate. Note that this is defined over
        transitions, not frames.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the solution. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.
    method : DGAMethod, optional
        Method for estimating the solution. If None (default), use
        ``DGA(lag)``.
    output : str, optional
        Type of output to return. The default ('projection') returns
        the projected solution.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimate of the solution of the forward Feynman-Kac formulat at
        each frame.

    """
    if method is None:
        method = DGA(lag)
    stat = ForwardVectorFeynmanKac(
        basis, weights, transitions, in_domain, function, guess, test_basis=test_basis
    )
    out = method.fit_transform(stat, output=output)
    return out


def backward_extended_committor(
    basis,
    weights,
    transitions,
    in_domain,
    guess,
    lag,
    test_basis=None,
    *,
    method=None,
    output="projection",
):
    """Estimate the backward extended committor using DGA.

    Parameters
    ----------
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.
    method : DGAMethod, optional
        Method for estimating the solution. If None (default), use
        ``DGA(lag)``.
    output : str, optional
        Type of output to return. The default ('projection') returns
        the projected solution.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimated backward extended committor at each frame.

    """
    return backward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        0.0,
        guess,
        lag,
        test_basis=test_basis,
        method=method,
        output=output,
    )


def backward_extended_mfpt(
    basis,
    weights,
    transitions,
    in_domain,
    guess,
    lag,
    test_basis=None,
    *,
    method=None,
    output="projection",
):
    """Estimate the backward mean first passage time using DGA.

    Parameters
    ----------
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the mean first passage time . Must be zero
        outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the mean first passage time . Must obey boundary
        conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.
    method : DGAMethod, optional
        Method for estimating the solution. If None (default), use
        ``DGA(lag)``.
    output : str, optional
        Type of output to return. The default ('projection') returns
        the projected solution.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimated backward mean first passage time at each frame.

    """
    return backward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        1.0,
        guess,
        lag,
        test_basis=test_basis,
        method=method,
        output=output,
    )


def backward_extended_feynman_kac(
    basis,
    weights,
    transitions,
    in_domain,
    function,
    guess,
    lag,
    test_basis=None,
    *,
    method=None,
    output="projection",
):
    """Solve the backward Feynman-Kac formula using DGA.

    Parameters
    ----------
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the solution to the Feynman-Kac formula.
        Must be zero outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    function : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Function to integrate. Note that this is defined over steps, not
        frames.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the solution. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.
    method : DGAMethod, optional
        Method for estimating the solution. If None (default), use
        ``DGA(lag)``.
    output : str, optional
        Type of output to return. The default ('projection') returns
        the projected solution.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimate of the solution of the backward Feynman-Kac formula at
        each frame.

    """
    if method is None:
        method = DGA(lag)
    weights = shift_weights(weights, lag)
    stat = BackwardVectorFeynmanKac(
        basis, weights, transitions, in_domain, function, guess, test_basis=test_basis
    )
    out = method.fit_transform(stat, output=output)
    return out
