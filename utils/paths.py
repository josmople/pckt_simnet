from os.path import *

from os import makedirs


def glob(pathname, *, recursive=False, unique=True, sort=True, sort_key=None, sort_reverse=False):
    from glob import glob as _glob

    if isinstance(pathname, str):  # If simple string
        values = _glob(pathname, recursive=recursive)
    else:  # Assume pathname is iterable
        assert all([isinstance(p, str) for p in pathname]), "pathname must be a str or List[str]"
        values = []
        for path in pathname:
            values += _glob(path, recursive=recursive)

        if unique:
            from collections import OrderedDict
            values = OrderedDict.fromkeys(values).keys()

    if sort:
        return sorted(values, key=sort_key, reverse=sort_reverse)
    return values
