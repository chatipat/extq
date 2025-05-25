import numpy as np

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
