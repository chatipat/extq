import numpy as np
import sklearn.cluster

from ._labels import _labels_to_basis
from ._labels import renumber_labels


def kmeans_labels(cvs, num, **kwargs):
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num, **kwargs)
    kmeans.fit(np.concatenate(cvs))
    indices = np.cumsum([len(v) for v in cvs])[:-1]
    return renumber_labels(np.split(kmeans.labels_, indices))


def kmeans1d_labels(cv, num, **kwargs):
    return kmeans_labels(_stack(cv), num, **kwargs)


def kmeans2d_labels(cv1, cv2, num, **kwargs):
    return kmeans_labels(_stack(cv1, cv2), num, **kwargs)


def kmeans3d_labels(cv1, cv2, cv3, num, **kwargs):
    return kmeans_labels(_stack(cv1, cv2, cv3), num, **kwargs)


def kmeans_basis(cvs, num, sparse=True, **kwargs):
    labels = kmeans_labels(cvs, num, **kwargs)
    num = _num(labels)
    basis = []
    for indices in labels:
        basis.append(_labels_to_basis(indices, num, sparse=sparse))
    return basis


def kmeans1d_basis(cv, num, sparse=True, **kwargs):
    return kmeans_basis(_stack(cv), num, sparse=sparse, **kwargs)


def kmeans2d_basis(cv1, cv2, num, sparse=True, **kwargs):
    return kmeans_basis(_stack(cv1, cv2), num, sparse=sparse, **kwargs)


def kmeans3d_basis(cv1, cv2, cv3, num, sparse=True, **kwargs):
    return kmeans_basis(_stack(cv1, cv2, cv3), num, sparse=sparse, **kwargs)


def kmeans_domain_basis(cvs, in_domain, num, sparse=True, **kwargs):
    cvs_d = [v[d] for v, d in zip(cvs, in_domain)]
    labels = kmeans_labels(cvs_d, num, **kwargs)
    num = _num(labels)
    basis = []
    for indices_d, d in zip(labels, in_domain):
        indices = np.empty(len(d), dtype=indices_d.dtype)
        indices[d] = indices_d
        basis.append(_labels_to_basis(indices, num, sparse=sparse, mask=d))
    return basis


def kmeans1d_domain_basis(cv, in_domain, num, sparse=True, **kwargs):
    return kmeans_domain_basis(
        _stack(cv), in_domain, num, sparse=sparse, **kwargs
    )


def kmeans2d_domain_basis(cv1, cv2, in_domain, num, sparse=True, **kwargs):
    return kmeans_domain_basis(
        _stack(cv1, cv2), in_domain, num, sparse=sparse, **kwargs
    )


def kmeans3d_domain_basis(
    cv1, cv2, cv3, in_domain, num, sparse=True, **kwargs
):
    return kmeans_domain_basis(
        _stack(cv1, cv2, cv3), in_domain, num, sparse=sparse, **kwargs
    )


def _num(labels):
    return max(np.max(indices) for indices in labels) + 1


def _stack(*cvs):
    return [np.stack(vs, axis=-1) for vs in zip(*cvs)]
