from .. import linalg


def identity(mats, mems=[]):
    return mats[0] + sum((s + 1) * mems[s] for s in range(len(mems)))


def generator(mats, mems=[]):
    return mats[1] - mats[0] + sum(mems[s] for s in range(len(mems)))


def memory(mats):
    inv = linalg.inv(mats[0])
    tmat = inv @ mats[1]
    temp = []
    mems = []
    for t in range(len(mats) - 2):
        mems.append(
            mats[t + 2]
            - mats[t + 1] @ tmat
            - sum(mats[t - s] @ temp[s] for s in range(t))
        )
        temp.append(inv @ mems[t])
    return mems


def extrapolate(mats, mems, lag):
    inv = linalg.inv(mats[0])
    tmat = inv @ mats[1]
    temp = [inv @ mem for mem in mems]
    xmats = []
    for t in range(lag + 1):
        if t < len(mats):
            xmats.append(mats[t])
        elif t < len(mems) + 2:
            xmats.append(
                xmats[t - 1] @ tmat
                + mems[t - 2]
                + sum(xmats[t - s - 2] @ temp[s] for s in range(t - 2))
            )
        else:
            xmats[t].append(
                xmats[t - 1] @ tmat
                + sum(xmats[t - s - 2] @ temp[s] for s in range(len(mems)))
            )
    return xmats
