import numba as nb
import numpy as np


def forward_fixed_time(trajs, lag):
    return [np.full(len(traj), lag) for traj in trajs]


def forward_fixed_distance(trajs, dist):
    return [_forward_fixed_distance(traj, dist) for traj in trajs]


@nb.njit
def _forward_fixed_distance(traj, dist):
    assert dist > 0
    n = len(traj)
    out = np.arange(n - 1, -1, -1)
    for t in range(n):
        for s in range(t + 1, n):
            if np.sum((traj[s] - traj[t]) ** 2) >= dist**2:
                out[t] = s - t
                break
    return out


def forward_first_switch(labels):
    return [_forward_first_switch(l) for l in labels]


@nb.njit
def _forward_first_switch(l):
    n = len(l)
    out = np.arange(n - 1, -1, -1)
    for t in range(n):
        for s in range(t + 1, n):
            if l[s] != l[t]:
                out[t] = s - t
                break
    return out


def forward_longest_time(trajs):
    return [np.arange(len(traj) - 1, -1, -1) for traj in trajs]
