from ._utils import badd
from ._utils import bmap
from ._utils import bshape
from ._utils import bsub
from ._utils import from_blocks
from ._utils import inv
from ._utils import to_blocks


def identity(bmats):
    return bmats[0]


def identity_with_memory(beye, bmems):
    result = beye
    for n, bmem in enumerate(bmems, start=1):
        result = badd(bmap(lambda a: a * n, bmem), result)
    return result


def generator(bmats):
    return bsub(bmats[1], bmats[0])


def generator_with_memory(bgen, bmems):
    result = bgen
    for bmem in bmems:
        result = badd(bmem, result)
    return result


def memory(bmats):
    shape = bshape(bmats[0])
    p = from_blocks(bmats[0], shape)
    pinv = inv(p)
    mats = [from_blocks(bmat, shape) @ pinv for bmat in bmats]
    mems = _memory(mats)
    bmems = [to_blocks(mem @ p, shape) for mem in mems]
    return bmems


def extrapolate(bmats, bmems, lag):
    shape = bshape(bmats[0])
    p = from_blocks(bmats[0], shape)
    pinv = inv(p)
    mats = [from_blocks(bmat, shape) @ pinv for bmat in bmats]
    mems = [from_blocks(bmem, shape) @ pinv for bmem in bmems]
    xmats = _extrapolate(mats, mems, lag)
    bxmats = [to_blocks(xmat @ p, shape) for xmat in xmats]
    return bxmats


def _memory(mats):
    mems = [None] * (len(mats) - 2)
    for t in range(len(mats) - 2):
        mems[t] = (
            mats[t + 2]
            - mats[t + 1] @ mats[1]
            - sum(mats[t - s] @ mems[s] for s in range(t))
        )
    return mems


def _extrapolate(mats, mems, lag):
    xmats = list(mats) + [None] * (lag + 1 - len(mats))
    for t in range(len(mats), lag + 1):
        xmats[t] = xmats[t - 1] @ xmats[1] + sum(
            xmats[t - s - 2] @ mems[s] for s in range(min(t - 1, len(mems)))
        )
    return xmats
