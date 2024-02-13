import warnings

import numpy as np
import scipy.sparse

__all__ = [
    "augment_generator",
    "build_kernel",
    "broadcast_kernel",
    "joint_kernel",
    "moveaxis_kernel",
    "spbroadcast",
    "spmoveaxis",
    "spouter",
]


def augment_generator(generator, spatial, temporal):
    """Augment the generator.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    spatial : (n_indices * n_points, n_indices * n_points) sparse matrix of float
        Spatial component of the augmenting transition kernel.
    temporal : (n_indices * n_points, n_indices * n_points) sparse matrix of float
        Temporal component of the augmenting transition kernel.

    Returns
    -------
    (n_indices * n_points, n_indices * n_points) sparse matrix of float
        Augmented generator matrix.

    """
    m = generator.shape[0]
    n = spatial.shape[0]
    assert n % m == 0
    gen_spatial, gen_temporal = broadcast_kernel(
        scipy.sparse.identity(m), generator, m, (n // m, m)
    )
    joint_spatial, joint_temporal = joint_kernel(
        gen_spatial, gen_temporal, spatial, temporal
    )
    if (
        (joint_spatial @ joint_spatial != joint_spatial).sum()
        or (joint_spatial @ joint_temporal != joint_temporal).sum()
        or (joint_temporal @ joint_spatial != joint_temporal).sum()
    ):
        warnings.warn(
            "zero lag time augmented transition matrix not an identity element"
        )
    return joint_spatial, joint_temporal


def build_kernel(generator, n_indices, entries):
    """Build an augmenting kernel at nonzero entries of the generator.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix
        Augmenting kernel is calculated at nonzero entries of this
        matrix. The values of the entries are not used otherwise.
    n_indices : int
        Dimension of the augmenting space.
    entries : callable
        Function to compute the entries of augmenting kernel. This is
        called as ``entries(row, col, spatial, temporal)``. Parameters
        ``row`` and ``col`` are ``(n_entries,)`` arrays of ``int`` and
        correspond to the row and column indices of `generator`.
        The function should write entries ``f[row, col]`` of the
        augmenting kernel to ``spatial`` and ``temporal``, which are
        zero-initialized ``(n_indices, n_indices, n_entries)`` arrays of
        ``float``.

    Returns
    -------
    (n_indices * n_points, n_indices * n_points) sparse matrix of float
        Augmenting kernel.

    """
    assert generator.shape[0] == generator.shape[1]

    # compute entries
    row, col = generator.nonzero()
    spatial_data = np.zeros((n_indices, n_indices, len(row)))
    temporal_data = np.zeros((n_indices, n_indices, len(row)))
    entries(row, col, spatial_data, temporal_data)

    # convert to block matrix
    spatial_blocks = np.empty((n_indices, n_indices), dtype=object)
    temporal_blocks = np.empty((n_indices, n_indices), dtype=object)
    for i in range(n_indices):
        for j in range(n_indices):
            spatial_blocks[i, j] = scipy.sparse.coo_matrix(
                (spatial_data[i, j], (row, col)), shape=generator.shape
            )
            temporal_blocks[i, j] = scipy.sparse.coo_matrix(
                (temporal_data[i, j], (row, col)), shape=generator.shape
            )
    spatial = scipy.sparse.bmat(spatial_blocks)
    temporal = scipy.sparse.bmat(temporal_blocks)

    return spatial, temporal


def broadcast_kernel(spatial, temporal, input_shape, output_shape):
    """Broadcast the transition kernel to a shape.

    Parameters
    ----------
    spatial : (input_size, input_size) sparse matrix of float
        Spatial component of the transition kernel.
    temporal : (input_size, input_size) sparse matrix of float
        Temporal component of the transition kernel.
    input_shape : int or sequence of int
        Shape of the input state space. The size of the input state
        space is ``input_size = numpy.product(input_shape)``.
    output_shape : int or sequence of int
        Shape of the output state space. The size of the output state
        space is ``output_size = numpy.product(output_shape)``.

    Returns
    -------
    spatial : (output_size, output_size) sparse matrix of float
        Spatial component of the broadcasted transition kernel.
    temporal : (output_size, output_size) sparse matrix of float
        Temporal component of the broadcasted transition kernel.

    """
    mat = spbroadcast(input_shape, output_shape)
    spatial = mat.T @ spatial @ mat
    temporal = mat.T @ temporal @ mat
    return spatial, temporal


def joint_kernel(spatial1, temporal1, spatial2, temporal2):
    """Returns the joint transition kernel.

    Parameters
    ----------
    spatial1, spatial2 : (n_points, n_points) sparse matrix of float
        Spatial component of each transition kernel.
    temporal1, temporal2 : (n_points, n_points) sparse matrix of float
        Temporal component of each transition kernel.

    Returns
    -------
    spatial : (n_points, n_points) sparse matrix of float
        Spatial component of the joint transition kernel.
    temporal : (n_points, n_points) sparse matrix of float
        Temporal component of the joint transition kernel.

    """
    spatial = spatial1.multiply(spatial2)
    temporal = temporal1.multiply(spatial2) + spatial1.multiply(temporal2)
    return spatial, temporal


def moveaxis_kernel(spatial, temporal, shape, source, destination):
    """Permute the order of state variables in the transition kernel.

    Parameters
    ----------
    spatial : (n_points, n_points) sparse matrix of float
        Spatial component of the transition kernel.
    temporal : (n_points, n_points) sparse matrix of float
        Temporal component of the transition kernel.
    shape : int or sequence of int
        Shape of the state space. The size of the state space is
        ``n_points = numpy.product(shape)``.
    source : int or sequence of int
        Initial indices of the state space variables to move.
    destination : int or sequence of int
        Final indices of the state space variables to move.

    Returns
    -------
    spatial : (n_points, n_points) sparse matrix of float
        Spatial component of the permuted transition kernel.
    temporal : (n_points, n_points) sparse matrix of float
        Temporal component of the permuted transition kernel.

    """
    mat = spmoveaxis(shape, source, destination)
    spatial = mat.T @ spatial @ mat
    temporal = mat.T @ temporal @ mat
    return spatial, temporal


def spbroadcast(input_shape, output_shape, dtype=np.float_):
    """Returns a matrix for broadcasting flattened dimensions.

    Parameters
    ----------
    input_shape : int or sequence of int
        Shape of the input dimensions. The size of the flattened input
        is ``input_size = numpy.product(input_shape)``.
    output_shape : int or sequence of int
        Shape of the output dimensions. The size of the flattened output
        is ``output_size = numpy.product(output_shape)``.
    dtype : data-type, optional
        Data-type of the output matrix.

    Returns
    -------
    (input_size, output_size) sparse matrix
        Matrix to transform the flattened input dimensions to the
        flattened output dimensions.

    """
    input_size = np.product(input_shape)
    output_size = np.product(output_shape)
    data = np.ones(output_size, dtype=dtype)
    rows = np.broadcast_to(
        np.arange(input_size).reshape(input_shape), output_shape
    ).reshape(output_size)
    cols = np.arange(output_size)
    return scipy.sparse.coo_matrix(
        (data, (rows, cols)), shape=(input_size, output_size)
    )


def spmoveaxis(shape, source, destination, dtype=np.float_):
    """Returns a matrix for permuting flattened dimensions.

    Parameters
    ----------
    shape : int or sequence of int
        Shape of the input dimensions. The size of the flattened
        dimensions is ``size = numpy.product(shape)``.
    source : int or sequence of int
        Initial indices of the dimensions.
    destination : int or sequence of int
        Final indices of the dimensions.
    dtype : data-type, optional
        Data-type of the output matrix.

    Returns
    -------
    (size, size) sparse matrix
        Matrix to permute the flattened dimensions.

    """
    size = np.product(shape)
    data = np.ones(size, dtype=dtype)
    rows = np.moveaxis(
        np.arange(size).reshape(shape), source, destination
    ).reshape(size)
    cols = np.arange(size)
    return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(size, size))


def spouter(m, op, a, b):
    """Compute the outer op of two vectors at nonzero entries of m.

    Parameters
    ----------
    m : sparse matrix
        Outer op is calculated at nonzero entries of this matrix.
    op : callable
        Ufunc taking in two vectors and returning one vector.
    a : array-like
        First input vector, flattened if not 1D.
    b : array-like
        Second input vector, flattened if not 1D. Must be the same shape
        as a.

    Returns
    -------
    sparse matrix
        Sparse matrix c with entries c[i,j] = op(a[i],b[j]) where
        m[i,j] is nonzero.

    """
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape
    row, col = m.nonzero()
    data = op(a.ravel()[row], b.ravel()[col])
    return scipy.sparse.csr_matrix((data, (row, col)), m.shape)
