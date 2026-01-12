import numpy as np
import scipy as sp
import einops

from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple
from .linalg import batched_block, batched_block_diag, batched_kron


class CTWFA:
    def __init__(
        self,
        initial,
        transitions,
        time_transitions,
        final,
        epsilon_transition=None,
        dtype=None,
    ):
        input_shape = transitions.shape[:-2]
        state_size = transitions.shape[-2]

        if epsilon_transition is None:
            epsilon_transition = np.zeros(
                (state_size, state_size), dtype=transitions.dtype
            )

        assert initial.shape == final.shape == (state_size,)
        assert (
            transitions.shape
            == time_transitions.shape
            == (*input_shape, state_size, state_size)
        )
        assert epsilon_transition.shape == (state_size, state_size)

        if dtype is None:
            dtype = np.result_type(
                initial, transitions, time_transitions, final, epsilon_transition
            )

        initial = initial.astype(dtype, copy=False)
        transitions = transitions.astype(dtype, copy=False)
        time_transitions = time_transitions.astype(dtype, copy=False)
        final = final.astype(dtype, copy=False)
        epsilon_transition = epsilon_transition.astype(dtype, copy=False)

        self.initial = initial
        self.transitions = transitions
        self.time_transitions = time_transitions
        self.final = final
        self.epsilon_transition = epsilon_transition

        self.input_shape = input_shape
        self.state_size = state_size
        self.dtype = dtype

    def __str__(self):
        fields = [
            f"state_size={self.state_size}",
            f"input_shape={self.input_shape}",
            f"dtype={self.dtype}",
        ]
        fields = ", ".join(fields)
        return f"CTWFA({fields})"

    def __repr__(self):
        def field_repr(key, value):
            return key + "=" + repr(value).replace("\n", "\n" + (len(key) + 7) * " ")

        fields = [
            field_repr("initial", self.initial),
            field_repr("transitions", self.transitions),
            field_repr("time_transitions", self.time_transitions),
            field_repr("final", self.final),
            field_repr("epsilon_transition", self.epsilon_transition),
        ]
        fields = ",\n      ".join(fields)
        return f"CTWFA({fields})"

    def __pos__(self):
        return positive(self)

    def __neg__(self):
        return negative(self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __invert__(self):
        return logical_not(self)

    def __and__(self, other):
        return logical_and(self, other)

    def __rand__(self, other):
        return logical_and(other, self)

    def __or__(self, other):
        return logical_or(self, other)

    def __ror__(self, other):
        return logical_or(other, self)


# ==================
# creation functions
# ==================


def _empty(state_size, input_shape, dtype):
    return CTWFA(
        np.zeros(state_size, dtype=dtype),
        np.zeros((*input_shape, state_size, state_size), dtype=dtype),
        np.zeros((*input_shape, state_size, state_size), dtype=dtype),
        np.zeros(state_size, dtype=dtype),
    )


def _scalar(value):
    # a(seq) == value
    out = _empty(1, (), value.dtype)
    out.initial[0] = value
    out.transitions[0, 0] = 1
    out.final[0] = 1
    return out


def asautomaton(obj):
    if isinstance(obj, CTWFA):
        return obj
    obj = np.array(obj)
    if np.ndim(obj) == 0 and np.issubdtype(obj.dtype, np.number):
        return _scalar(obj)
    raise TypeError(f"Cannot convert type {type(obj)} to CTWFA.")


def empty(input_shape=(), dtype=float):
    # a(seq) = 0
    return _empty(1, input_shape, dtype)


def zero(input_shape=(), dtype=float):
    # a([]) == 1
    # a(seq) == 0 if len(seq) > 0
    out = _empty(1, input_shape, dtype)
    out.initial[0] = 1
    out.final[0] = 1
    return out


def one_event(weights, time_weights):
    # a([i]) == weights[i]
    assert weights.shape == time_weights.shape
    dtype = np.result_type(weights, time_weights)
    out = _empty(2, weights.shape, dtype)
    out.initial[0] = 1
    out.transitions[..., 0, 1] = weights
    out.time_transitions[..., 0, 1] = time_weights
    out.final[1] = 1
    return prune(out)


def one_hot(index, input_size, dtype=float):
    out = _empty(2, [input_size], dtype)
    out.initial[0] = 1
    out.transitions[index, 0, 1] = 1
    out.final[1] = 1
    return out


# ======================
# manipulation functions
# ======================


def moveaxis(a, source, destination):
    ndim = len(a.input_shape)
    source = normalize_axis_tuple(source, ndim)
    destination = normalize_axis_tuple(destination, ndim)
    return CTWFA(
        a.initial,
        np.moveaxis(a.transitions, source, destination),
        np.moveaxis(a.time_transitions, source, destination),
        a.final,
        a.epsilon_transition,
    )


def reshape(a, input_shape):
    state_size = a.state_size
    shape = (*input_shape, state_size, state_size)
    return CTWFA(
        a.initial,
        a.transitions.reshape(shape),
        a.time_transitions.reshape(shape),
        a.final,
        a.epsilon_transition,
    )


def squeeze(a, axis):
    input_shape = a.input_shape
    ndim = len(input_shape)
    axis = normalize_axis_index(axis, ndim)
    assert input_shape[axis] == 1
    input_shape = input_shape[:axis] + input_shape[axis + 1 :]
    return reshape(a, input_shape)


def unsqueeze(a, axis):
    input_shape = a.input_shape
    ndim = len(input_shape) + 1
    axis = normalize_axis_index(axis, ndim)
    input_shape = input_shape[:axis] + (1,) + input_shape[axis:]
    return reshape(a, input_shape)


def broadcast_to(a, input_shape):
    state_size = a.state_size
    shape = (*input_shape, state_size, state_size)
    return CTWFA(
        a.initial,
        np.broadcast_to(a.transitions, shape),
        np.broadcast_to(a.time_transitions, shape),
        a.final,
        a.epsilon_transition,
    )


# =====================
# arithmetic operations
# =====================


def positive(a):
    return asautomaton(a)


def negative(a):
    return multiply(-1, a)


def add(a, b):
    a = asautomaton(a)
    b = asautomaton(b)
    out = CTWFA(
        np.concatenate([a.initial, b.initial]),
        batched_block_diag(a.transitions, b.transitions),
        batched_block_diag(a.time_transitions, b.time_transitions),
        np.concatenate([a.final, b.final]),
        batched_block_diag(a.epsilon_transition, b.epsilon_transition),
    )
    return prune(out)


def subtract(a, b):
    return add(a, multiply(-1, b))


def multiply(a, b):
    a = asautomaton(a)
    b = asautomaton(b)
    out = CTWFA(
        np.kron(a.initial, b.initial),
        batched_kron(a.transitions, b.transitions),
        batched_kron(a.transitions, b.time_transitions)
        + batched_kron(a.time_transitions, b.transitions),
        np.kron(a.final, b.final),
        np.kron(a.epsilon_transition, b.epsilon_transition),
    )
    return prune(out)


def sum(automatons):
    out = asautomaton(0)
    for a in automatons:
        out = add(out, a)
    return out


def prod(automatons):
    out = asautomaton(1)
    for a in automatons:
        out = multiply(out, a)
    return out


# ==============
# linear algebra
# ==============


def trace(a):
    a = asautomaton(a)

    def f(a):
        return np.einsum("...iiab->...ab", a)

    out = CTWFA(
        a.initial,
        f(a.transitions),
        f(a.time_transitions),
        a.final,
        a.epsilon_transition,
    )
    return prune(out)


def vecdot(a, b):
    return _contract("...iab,...icd->...acbd", a, b)


def vecmat(a, b):
    return _contract("...iab,...ijcd->...jacbd", a, b)


def matvec(a, b):
    return _contract("...ijab,...jcd->...iacbd", a, b)


def matmul(a, b):
    return _contract("...ijab,...jkcd->...ikacbd", a, b)


def outer(a, b):
    return _contract("...iab,...jcd->...ijacbd", a, b)


def _contract(subscripts, a, b):
    a = asautomaton(a)
    b = asautomaton(b)

    def f(x, y):
        z = np.einsum(subscripts, x, y)
        return z.reshape(
            *z.shape[:-4], z.shape[-4] * z.shape[-3], z.shape[-2] * z.shape[-1]
        )

    out = CTWFA(
        np.kron(a.initial, b.initial),
        f(a.transitions, b.transitions),
        f(a.transitions, b.time_transitions) + f(a.time_transitions, b.transitions),
        np.kron(a.final, b.final),
        np.kron(a.epsilon_transition, b.epsilon_transition),
    )
    return prune(out)


# ==================
# logical operations
# ==================


def logical_not(a):
    return 1 - a


def logical_and(a, b):
    return a * b


def logical_or(a, b):
    return a + b - a * b


def is_bool(a, rcond=None):
    b = a * (1 - a)
    b = optimize(b, rcond=rcond)
    return b.state_size == 0


# ===================
# regular expressions
# ===================


def zero_or_one(a):
    return zero(dtype=a.dtype) + a


def zero_or_more(a):
    return zero_or_one(one_or_more(a))


def one_or_more(a):
    out = CTWFA(
        a.initial,
        a.transitions,
        a.time_transitions,
        a.final,
        a.epsilon_transition + np.outer(a.final, a.initial),
    )
    return prune(out)


def concatenate(automatons):
    n = len(automatons[0].transitions)
    assert all(len(a.transitions) == n for a in automatons)
    initial = np.concatenate(
        [automatons[0].initial] + [np.zeros_like(a.initial) for a in automatons[1:]]
    )
    transitions = batched_block_diag(*[a.transitions for a in automatons])
    time_transitions = batched_block_diag(*[a.time_transitions for a in automatons])
    final = np.concatenate(
        [np.zeros_like(a.final) for a in automatons[:-1]] + [automatons[-1].final]
    )
    epsilon_transition = np.full((len(automatons), len(automatons)), None)
    for i in range(len(automatons)):
        epsilon_transition[i, i] = automatons[i].epsilon_transition
    for i in range(len(automatons) - 1):
        epsilon_transition[i, i + 1] = np.outer(
            automatons[i].final, automatons[i + 1].initial
        )
    epsilon_transition = batched_block(epsilon_transition)
    out = CTWFA(initial, transitions, time_transitions, final, epsilon_transition)
    return prune(out)


# =================
# utility functions
# =================


class Kernel:
    def __init__(self, full, initial, middle, final):
        self.full = full
        self.initial = initial
        self.middle = middle
        self.final = final


def ctwfa_to_kernel(a, dt=1.0, rcond=None):
    a = eliminate_epsilon_transition(a)

    m0 = np.zeros((*a.input_shape, a.state_size + 2, a.state_size + 2), dtype=a.dtype)
    m0[..., 0, 0] = 1
    m0[..., 0, 1:-1] = a.initial
    m0[..., 1:-1, 1:-1] = a.transitions
    m0[..., 1:-1, -1] = a.final
    m0[..., -1, -1] = 1

    m1 = np.zeros((*a.input_shape, a.state_size + 2, a.state_size + 2), dtype=a.dtype)
    m1[..., 1:-1, 1:-1] = a.time_transitions * dt

    m = np.zeros((*a.input_shape, a.state_size + 2, a.state_size + 2), dtype=a.dtype)
    for i in np.ndindex(a.input_shape):
        m[i] = _gen_expm(m0[i], m1[i], rcond=rcond)

    full = m[..., 0, -1]
    initial = m[..., 0, 1:-1]
    middle = m[..., 1:-1, 1:-1]
    final = m[..., 1:-1, -1]
    return Kernel(full, initial, middle, final)


def _gen_expm(a, b, rcond=None):
    n, _ = a.shape
    assert a.shape == b.shape == (n, n)
    U, s, Vh = sp.linalg.svd(a - np.identity(n))
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * len(s)
    tol = np.max(s, initial=0.0) * rcond
    num = np.sum(s > tol)
    U = U[:, num:]
    Vh = Vh[num:, :]
    Uh = U.conj().T
    V = Vh.conj().T
    # P = V @ sp.linalg.solve(Uh @ V, Uh)
    Vh_P = sp.linalg.solve(Uh @ V, Uh)
    return V @ sp.linalg.expm(Vh_P @ b @ V) @ Vh_P


def prune(a):
    if a.state_size == 0:
        return a

    initial = a.initial
    all_transitions, ps = einops.pack(
        [a.transitions, a.time_transitions, a.epsilon_transition], "* i j"
    )
    final = a.final

    while True:
        # remove states with all-zero rows and/or columns
        nonzero_rows = np.any(all_transitions, axis=(0, 2)) | final.astype(bool)
        nonzero_cols = np.any(all_transitions, axis=(0, 1)) | initial.astype(bool)
        mask = nonzero_rows & nonzero_cols
        if np.all(mask):
            break
        initial = initial[mask]
        all_transitions = all_transitions[:, *np.ix_(mask, mask)]
        final = final[mask]

    transitions, time_transitions, epsilon_transition = einops.unpack(
        all_transitions, ps, "* i j"
    )
    return CTWFA(initial, transitions, time_transitions, final, epsilon_transition)


def eliminate_epsilon_transition(a):
    # x @ eps_star == x + x @ eps + x @ eps @ eps + ...
    eps = a.epsilon_transition
    eps_star = np.linalg.inv(np.identity(len(eps), dtype=eps.dtype) - eps)
    initial = a.initial @ eps_star
    transitions = a.transitions @ eps_star
    time_transitions = a.time_transitions @ eps_star
    final = a.final
    return CTWFA(initial, transitions, time_transitions, final)


def optimize(a, rcond=None):
    a = prune(eliminate_epsilon_transition(a))
    if a.state_size == 0:
        return a

    state_size = a.state_size
    while True:
        # TODO: find why multiple iterations are needed
        a = prune(_optimize(a, rcond=rcond))
        if a.state_size == 0 or a.state_size == state_size:
            return a
        state_size = a.state_size


def _optimize(a, rcond=None):
    a = eliminate_epsilon_transition(a)
    initial = a.initial
    transitions = a.transitions
    time_transitions = a.time_transitions
    final = a.final

    # independent basis of transition matrices
    mats, _ = einops.pack([transitions, time_transitions], "* i j")
    mats = mats.reshape(len(mats), a.state_size * a.state_size)
    mats = _remove_dependent_columns(mats.T, rcond=rcond).T
    mats = mats.reshape(len(mats), a.state_size, a.state_size)

    # accessible span of rows and columns
    row_span = _multi_krylov_span(mats, final[:, None], rcond=rcond)
    col_span = _multi_krylov_span(
        np.moveaxis(mats, 1, 2), initial[:, None], rcond=rcond
    )

    # orthonormalize to make row_proj and col_proj more balanced
    row_span, _ = sp.linalg.qr(row_span, mode="economic")
    col_span, _ = sp.linalg.qr(col_span, mode="economic")

    # intersection of row and column spans
    cov = row_span.T @ col_span
    u, s, vh = sp.linalg.svd(cov, full_matrices=False)
    if rcond is None:
        rcond = np.max(cov.shape) * np.finfo(s.dtype).eps
    tol = np.max(s, initial=0.0) * rcond
    num = np.sum(s > tol)
    # col_proj @ row_proj.T is a projection matrix
    col_proj = row_span @ (u[:, :num] / np.sqrt(s[:num]))
    row_proj = col_span @ (vh.T[:, :num] / np.sqrt(s[:num]))

    initial = initial @ col_proj
    transitions = row_proj.T @ transitions @ col_proj
    time_transitions = row_proj.T @ time_transitions @ col_proj
    final = final @ row_proj
    return CTWFA(initial, transitions, time_transitions, final)


def _multi_krylov_span(mats, basis, rcond=None):
    n, k, _ = mats.shape
    rank = 0
    while basis.shape[1] != rank:
        rank = basis.shape[1]
        image_basis = np.moveaxis(mats @ basis, 1, 0)
        image_basis = image_basis.reshape(k, n * rank)
        basis = np.concatenate([basis, image_basis], axis=1)
        basis = _remove_dependent_columns(basis, rcond=rcond)
    return basis


def _remove_dependent_columns(a, rcond=None):
    r, p = sp.linalg.qr(a, mode="r", pivoting=True)
    if rcond is None:
        rcond = np.max(a.shape) * np.finfo(r.dtype).eps
    tol = np.max(np.abs(a), initial=0.0) * rcond
    rank = np.sum(np.abs(np.diag(r)) > tol)
    return a[:, p[:rank]]
