import numpy as np


class Automaton:
    def __init__(self, initial, transitions, final):
        n, k = transitions.shape[:2]
        assert initial.shape == final.shape == (k,)
        assert transitions.shape == (n, k, k)

        self.initial = initial
        self.transitions = transitions
        self.final = final

    def __add__(self, other):
        return sum([self, other])

    def __mul__(self, other):
        assert len(self.transitions) == len(other.transitions)
        return Automaton(
            np.kron(self.initial, other.initial),
            _batched_kron(self.transitions, other.transitions),
            np.kron(self.final, other.final),
        )

    def __call__(self, events, axis=-1):
        out = self.initial[None, :]
        for i in np.moveaxis(events, axis, 0):
            out = out @ self.transitions[i]
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
    assert automaton.initial @ automaton.final == 0  # automaton([]) == 0
    # zero([]) == 1
    # zero(seq) == 0 if len(seq) > 0
    zero = Automaton(
        np.ones(1, dtype=automaton.initial.dtype),
        np.zeros((len(automaton.transitions), 1, 1), dtype=automaton.transitions.dtype),
        np.ones(1, dtype=automaton.final.dtype),
    )
    return zero + automaton


def one_or_more(automaton):
    assert automaton.initial @ automaton.final == 0  # automaton([]) == 0
    # epsilon transitions = automaton.final[:, None] @ automaton.initial[None, :]
    # more than one epsilon transition == 0
    return Automaton(
        automaton.initial,
        automaton.transitions
        + automaton.transitions @ automaton.final[:, None] @ automaton.initial[None, :],
        automaton.final,
    )


def zero_or_more(automaton):
    return zero_or_one(one_or_more(automaton))


def scale(automaton, scale):
    return Automaton(automaton.initial * scale, automaton.transitions, automaton.final)


def sum(automatons):
    n = len(automatons[0].transitions)
    assert all(len(a.transitions) == n for a in automatons)
    initial = np.concatenate([a.initial for a in automatons])
    transitions = _batched_block_diag(*[a.transitions for a in automatons])
    final = np.concatenate([a.final for a in automatons])
    return Automaton(initial, transitions, final)


def prod(automatons):
    out = automatons[0]
    for a in automatons[1:]:
        out = out * a
    return out


def concatenate(automatons):
    n = len(automatons[0].transitions)
    assert all(len(a.transitions) == n for a in automatons)

    # initial = automatons[0].initial
    # final = automatons[-1].final
    # transitions = automatons[:].transitions + epsilon transitions
    # epsilon transitions =
    #   automatons[:-1].final[:, None] @ automatons[1:].initial[None, :]
    # absorb epsilon transitions into the matrix to the left

    initial = [automatons[0].initial]
    for i in range(1, len(automatons)):
        initial.append(initial[i - 1] @ automatons[i - 1].final * automatons[i].initial)
    initial = np.concatenate(initial)

    transitions = np.full((len(automatons), len(automatons)), None)
    for i in range(len(automatons)):
        transitions[i, i] = automatons[i].transitions
        for j in range(i + 1, len(automatons)):
            transitions[i, j] = (
                transitions[i, j - 1]
                @ automatons[j - 1].final[:, None]
                @ automatons[j].initial[None, :]
            )
    transitions = _block(transitions)

    final = np.concatenate(
        [np.zeros_like(a.final) for a in automatons[:-1]] + [automatons[-1].final]
    )

    return Automaton(initial, transitions, final)


def prune(automaton):
    initial = automaton.initial
    transitions = automaton.transitions
    final = automaton.final
    while True:
        # remove states with all-zero rows and/or columns
        nonzero_rows = np.any(transitions, axis=(0, 2)) | final.astype(bool)
        nonzero_cols = np.any(transitions, axis=(0, 1)) | initial.astype(bool)
        mask = nonzero_rows & nonzero_cols
        if np.all(mask):
            return Automaton(initial, transitions, final)
        initial = initial[mask]
        transitions = transitions[:, mask, :][:, :, mask]
        final = final[mask]


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
