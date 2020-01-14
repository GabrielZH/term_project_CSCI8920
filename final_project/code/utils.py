

def fs2dict(fs):
    return dict(fs)


def dict2fs(dic):
    return frozenset(dic.items())


def list2tuple(ls):
    tp = tuple(tuple(x) for x in ls)
    return tp


def tuple2list(tp):
    ls = list(list(x) for x in tp)
    return ls
