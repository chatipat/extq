import numpy as np
import scipy as sp

from . import linalg


class Automaton:
    def __init__(self, initial, transitions, final, epsilon_transition=None):
        n, k = transitions.shape[:2]
        if epsilon_transition is None:
            epsilon_transition = np.zeros((k, k), dtype=transitions.dtype)

        assert initial.shape == final.shape == (k,)
        assert transitions.shape == (n, k, k)
        assert epsilon_transition.shape == (k, k)

        self.initial = initial
        self.transitions = transitions
        self.final = final
        self.epsilon_transition = epsilon_transition

    def __add__(self, other):
        return sum([self, other])

    def __mul__(self, other):
        assert len(self.transitions) == len(other.transitions)
        return Automaton(
            np.kron(self.initial, other.initial),
            linalg.batched_kron(self.transitions, other.transitions),
            np.kron(self.final, other.final),
            np.kron(self.epsilon_transition, other.epsilon_transition),
        )

    def __call__(self, events, axis=-1):
        # out @ eps_star == out + out @ eps + out @ eps @ eps + ...
        eps = self.epsilon_transition
        eps_star = np.linalg.inv(np.identity(len(eps), dtype=eps.dtype) - eps)

        out = self.initial[None, :] @ eps_star
        for i in np.moveaxis(events, axis, 0):
            out = out @ self.transitions[i] @ eps_star
        out = out @ self.final[:, None]
        out = out.reshape(out.shape[:-2])
        return out


def one_event(weights):
    initial = np.zeros(2, dtype=weights.dtype)
    transitions = np.zeros((len(weights), 2, 2), dtype=weights.dtype)
    final = np.zeros(2, dtype=weights.dtype)
    initial[0] = 1
    transitions[:, 0, 1] = weights
    final[1] = 1
    return Automaton(initial, transitions, final)


def zero_or_one(automaton):
    # zero([]) == 1
    # zero(seq) == 0 if len(seq) > 0
    zero = Automaton(
        np.ones(1, dtype=automaton.initial.dtype),
        np.zeros((len(automaton.transitions), 1, 1), dtype=automaton.transitions.dtype),
        np.ones(1, dtype=automaton.final.dtype),
    )
    return zero + automaton


def one_or_more(automaton):
    return Automaton(
        automaton.initial,
        automaton.transitions,
        automaton.final,
        automaton.epsilon_transition + np.outer(automaton.final, automaton.initial),
    )


def zero_or_more(automaton):
    return zero_or_one(one_or_more(automaton))


def scale(automaton, scale):
    return Automaton(
        automaton.initial * scale,
        automaton.transitions,
        automaton.final,
        automaton.epsilon_transition,
    )


def sum(automatons):
    n = len(automatons[0].transitions)
    assert all(len(a.transitions) == n for a in automatons)
    initial = np.concatenate([a.initial for a in automatons])
    transitions = linalg.batched_block_diag(*[a.transitions for a in automatons])
    final = np.concatenate([a.final for a in automatons])
    epsilon_transition = linalg.batched_block_diag(
        *[a.epsilon_transition for a in automatons]
    )
    return Automaton(initial, transitions, final, epsilon_transition)


def prod(automatons):
    out = automatons[0]
    for a in automatons[1:]:
        out = out * a
    return out


def concatenate(automatons):
    n = len(automatons[0].transitions)
    assert all(len(a.transitions) == n for a in automatons)
    initial = np.concatenate(
        [automatons[0].initial] + [np.zeros_like(a.initial) for a in automatons[1:]]
    )
    transitions = linalg.batched_block_diag(*[a.transitions for a in automatons])
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
    epsilon_transition = linalg.batched_block(epsilon_transition)
    return Automaton(initial, transitions, final, epsilon_transition)


def prune(automaton):
    initial = automaton.initial
    transitions = automaton.transitions
    final = automaton.final
    epsilon_transition = automaton.epsilon_transition
    while True:
        # remove states with all-zero rows and/or columns
        nonzero_rows = (
            np.any(transitions, axis=(0, 2))
            | np.any(epsilon_transition, axis=1)
            | final.astype(bool)
        )
        nonzero_cols = (
            np.any(transitions, axis=(0, 1))
            | np.any(epsilon_transition, axis=0)
            | initial.astype(bool)
        )
        mask = nonzero_rows & nonzero_cols
        if np.all(mask):
            return Automaton(initial, transitions, final, epsilon_transition)
        initial = initial[mask]
        transitions = transitions[:, mask, :][:, :, mask]
        final = final[mask]
        epsilon_transition = epsilon_transition[mask, :][:, mask]


def eliminate_epsilon_transition(automaton):
    eps = automaton.epsilon_transition
    eps_star = np.linalg.inv(np.identity(len(eps), dtype=eps.dtype) - eps)
    initial = automaton.initial @ eps_star
    transitions = automaton.transitions @ eps_star
    final = automaton.final
    return Automaton(initial, transitions, final)


def optimize(automaton, rcond=None):
    automaton = eliminate_epsilon_transition(automaton)
    initial = automaton.initial
    transitions = automaton.transitions
    final = automaton.final

    # accessible span of rows and columns
    row_span = _multi_krylov_span(transitions, final[:, None], rcond=rcond)
    col_span = _multi_krylov_span(
        np.moveaxis(transitions, 1, 2), initial[:, None], rcond=rcond
    )

    # orthonormalize to make row_proj and col_proj more balanced
    row_span, _ = sp.linalg.qr(row_span, mode="economic")
    col_span, _ = sp.linalg.qr(col_span, mode="economic")

    # intersection of row and column spans
    cov = row_span.T @ col_span
    u, s, vh = sp.linalg.svd(cov, full_matrices=False)
    if rcond is None:
        rcond = np.max(cov.shape) * np.finfo(s.dtype).eps
    tol = np.max(s) * rcond
    num = np.sum(s > tol)
    # col_proj @ row_proj.T is a projection matrix
    col_proj = row_span @ (u[:, :num] / np.sqrt(s[:num]))
    row_proj = col_span @ (vh.T[:, :num] / np.sqrt(s[:num]))

    initial = initial @ col_proj
    transitions = row_proj.T @ transitions @ col_proj
    final = final @ row_proj
    return Automaton(initial, transitions, final)


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
