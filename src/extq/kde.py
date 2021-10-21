import numpy as np
import scipy.ndimage


def kdesum2d(
    x,
    y,
    w,
    *,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    xstd=None,
    ystd=None,
    nx=100,
    ny=100,
    cut=4.0,
):
    """Compute a 2D kernel density estimate.

    This function histograms the data, then uses a Gaussian filter to
    approximate a kernel density estimate with a Gaussian kernel.

    Parameters
    ----------
    x, y : ndarray or list/tuple of ndarray
        Coordinates of each frame.
    w : ndarray or list/tuple of ndarray
        Weight or value of each frame. The output is the sum of these
        values in each bin, after smoothing.
    xmin, xmax, ymin, ymax : float, optional
        Limits of kernel density estimate. If None, takes the min/max
        of the data along the coordinate.
    xstd, ystd : float, optional
        Standard deviation of the Gaussian filter. If None, these are
        set to (xmax - xmin) / nx and (ymax - ymin) / ny, respectively.
        Increase this to smooth the results more.
    nx, ny : int, optional
        Number of bins in each dimension. This should be set as high as
        reasonable, since xstd/ystd takes care of the smoothing.
    cut : float, optional
        Number of standard deviations at which to truncate the Gaussian
        filter. The default, 4, usually doesn't need to be changed.

    Returns
    -------
    kde : (nx, ny) ndarray
        Kernel density estimate, given as bins.
    xedges : (nx+1,) ndarray
        Bin edges along the x dimension.
    yedges : (ny+1,) ndarray
        Bin edges along the y dimension.

    """

    # flatten input to 1D arrays
    x = _flatten(x)
    y = _flatten(y)
    w = _flatten(w)

    # limits
    _xmin = np.min(x)
    _xmax = np.max(x)
    _ymin = np.min(y)
    _ymax = np.max(y)
    if xmin is None:
        xmin = _xmin
    if xmax is None:
        xmax = _xmax
    if ymin is None:
        ymin = _ymin
    if ymax is None:
        ymax = _ymax

    # separation between grid points
    xsep = (xmax - xmin) / nx
    ysep = (ymax - ymin) / ny

    # number of grid points to pad the boundaries,
    # since the Gaussian filter extends beyond the boundaries
    # usually overestimates the padding, but whatever
    ax = max(0, int(np.ceil((xmin - _xmin) / xsep + 1e-6)))
    bx = max(0, int(np.ceil((_xmax - xmax) / xsep + 1e-6)))
    ay = max(0, int(np.ceil((ymin - _ymin) / ysep + 1e-6)))
    by = max(0, int(np.ceil((_ymax - ymax) / ysep + 1e-6)))

    # output bin edges
    xedges = np.linspace(xmin, xmax, nx + 1)
    yedges = np.linspace(ymin, ymax, ny + 1)

    # bin edges, with the added padding
    xedges_padded = np.concatenate(
        [
            xmin + xsep * np.arange(-ax, 0),
            xedges,
            xmax + xsep * np.arange(1, bx + 1),
        ]
    )
    yedges_padded = np.concatenate(
        [
            ymin + ysep * np.arange(-ay, 0),
            yedges,
            ymax + ysep * np.arange(1, by + 1),
        ]
    )
    assert np.allclose(xedges_padded[1:] - xedges_padded[:-1], xsep)
    assert np.allclose(yedges_padded[1:] - yedges_padded[:-1], ysep)
    assert xedges_padded[0] <= _xmin and _xmax <= xedges_padded[-1]
    assert yedges_padded[0] <= _ymin and _ymax <= yedges_padded[-1]

    # construct 2D histogram on padded edges
    hist_padded, _, _ = np.histogram2d(
        x, y, weights=w, bins=(xedges_padded, yedges_padded)
    )

    # Gaussian kernel parameters
    if xstd is None:
        xstd = xsep
    if ystd is None:
        ystd = ysep

    # apply Gaussian filter to histogram
    kde_padded = scipy.ndimage.gaussian_filter(
        hist_padded,
        sigma=(xstd / xsep, ystd / ysep),  # in units of grid points
        mode="constant",
        truncate=cut,
    )

    # remove the padding
    assert ax + nx + bx == kde_padded.shape[0]
    assert ay + ny + by == kde_padded.shape[1]
    kde = kde_padded[ax : ax + nx, ay : ay + ny]

    return kde, xedges, yedges


def _flatten(a):
    if isinstance(a, np.ndarray):
        # avoid creating a new array (and using twice the memory)
        return np.ravel(a)
    else:
        return np.ravel(np.concatenate(a))
