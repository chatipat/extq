import numpy as np


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
            _batched_kron(self.transitions, other.transitions),
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
    transitions = _batched_block_diag(*[a.transitions for a in automatons])
    final = np.concatenate([a.final for a in automatons])
    epsilon_transition = _batched_block_diag(
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
    transitions = _batched_block_diag(*[a.transitions for a in automatons])
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
    epsilon_transition = _block(epsilon_transition)
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
            return Automaton(initial, transitions, final)
        initial = initial[mask]
        transitions = transitions[:, mask, :][:, :, mask]
        final = final[mask]
        epsilon_transition = epsilon_transition[mask, :][:, mask]


def _block(blocks):
    n1 = len(blocks)
    n2 = len(blocks[0])
    assert all(len(b) == n2 for b in blocks)

    dtypes = []
    k1 = [None] * n1
    k2 = [None] * n2
    for i in range(n1):
        for j in range(n2):
            a = blocks[i][j]
            if a is not None:
                dtypes.append(a.dtype)
                if k1[i] is None:
                    k1[i] = a.shape[-2]
                if k2[j] is None:
                    k2[j] = a.shape[-1]
                assert k1[i] == a.shape[-2]
                assert k2[j] == a.shape[-1]
    dtype = np.result_type(*dtypes)
    assert None not in k1
    assert None not in k2
    k1 = np.array(k1)
    k2 = np.array(k2)

    offset1 = np.concatenate([[0], np.cumsum(k1)])
    offset2 = np.concatenate([[0], np.cumsum(k2)])
    out = np.zeros((offset1[-1], offset2[-1]), dtype=dtype)
    for i in range(n1):
        for j in range(n2):
            a = blocks[i][j]
            if a is not None:
                out[offset1[i] : offset1[i + 1], offset2[j] : offset2[j + 1]] = a
    return out


def _batched_block_diag(*arrs):
    assert all(a.ndim >= 2 for a in arrs)
    shape = np.broadcast_shapes(*[a.shape[:-2] for a in arrs])
    k1 = np.sum([a.shape[-2] for a in arrs])
    k2 = np.sum([a.shape[-1] for a in arrs])
    dtype = np.result_type(*[a.dtype for a in arrs])

    out = np.zeros((*shape, k1, k2), dtype=dtype)
    r = 0
    c = 0
    for a in arrs:
        out[..., r : r + a.shape[-2], c : c + a.shape[-1]] = a
        r += a.shape[-2]
        c += a.shape[-1]
    assert r == k1 and c == k2

    return out


def _batched_kron(arr1, arr2):
    m1, n1 = arr1.shape[-2:]
    m2, n2 = arr2.shape[-2:]
    out = arr1[..., :, None, :, None] * arr2[..., None, :, None, :]
    out = out.reshape(*out.shape[:-4], m1 * m2, n1 * n2)
    return out
