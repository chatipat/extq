from .. import linalg
from ._utils import badd
from ._utils import bmap
from ._utils import bshape
from ._utils import bsub
from ._utils import from_blocks
from ._utils import to_blocks


def identity(bmats, bmems=None):
    result = bmats[0]
    if bmems is not None:
        for n, bmem in enumerate(bmems, start=1):
            result = badd(bmap(lambda a: a * n, bmem), result)
    return result


def generator(bmats, bmems=None):
    result = bsub(bmats[1], bmats[0])
    if bmems is not None:
        for bmem in bmems:
            result = badd(bmem, result)
    return result


def memory(bmats):
    shape = bshape(bmats[0])
    mats = [from_blocks(bmat, shape) for bmat in bmats]
    mems = _memory(mats)
    bmems = [to_blocks(mem, shape) for mem in mems]
    return bmems


def extrapolate(bmats, bmems, lag):
    shape = bshape(bmats[0])
    mats = [from_blocks(bmat, shape) for bmat in bmats]
    mems = [from_blocks(bmem, shape) for bmem in bmems]
    xmats = _extrapolate(mats, mems, lag)
    bxmats = [to_blocks(xmat, shape) for xmat in xmats]
    return bxmats


def _memory(mats):
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


def _extrapolate(mats, mems, lag):
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
