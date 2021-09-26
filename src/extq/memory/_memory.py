def memory(c):
    k = [None] * len(c)
    for t in range(1, len(c)):
        k[t] = c[t].copy()
        for s in range(1, t):
            k[t] -= k[s] @ c[t - s]
    return k


def extrapolate(c, k, lag):
    result = [c[0]] + [None] * lag
    for t in range(1, lag + 1):
        if t < len(c):
            result[t] = c[t]
        elif t < len(k):
            result[t] = k[t].copy()
            for s in range(1, t):
                result[t] += k[s] @ result[t - s]
        else:
            result[t] = 0.0
            for s in range(1, len(k)):
                result[t] += k[s] @ result[t - s]
    return result
