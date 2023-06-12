import numpy as np
import sklearn.metrics


def voronoi_labels(cvs, centers):
    """
    Label frames by the nearest center.

    Parameters
    ----------
    cvs : sequence of (n_frames[i], n_cvs) ndarray of float
        Collective variable at each frame.
    centers : (num, n_cvs) ndarray of float
        Cluster centers.

    Returns
    -------
    labels : list of (n_frames[i],) ndarray of int
        Index of the nearest center at each frame.

    """
    return [
        sklearn.metrics.pairwise_distances_argmin(vs, centers) for vs in cvs
    ]
