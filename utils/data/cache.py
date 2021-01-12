class Cache:

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __setitem__(self, idx, val):
        raise NotImplementedError()

    def __contains__(self, idx):
        raise NotImplementedError()


class DictCache(Cache):

    def __init__(self, data=None):
        self.data = {}
        self.data.update(data or {})

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, val):
        self.data[idx] = val

    def __contains__(self, idx):
        return idx in self.data


class LambdaCache(Cache):

    __slots__ = ["save_fn", "load_fn", "exist_fn"]

    def __init__(self, save_fn, load_fn, exist_fn):
        assert callable(save_fn)
        assert callable(load_fn)
        assert callable(exist_fn)

        self.save_fn = save_fn
        self.load_fn = load_fn
        self.exist_fn = exist_fn

    def __getitem__(self, idx):
        return self.load_fn(idx)

    def __setitem__(self, idx, val):
        self.save_fn(idx, val)

    def __contains__(self, idx):
        return self.exist_fn(idx)


class FileCache(Cache):

    __slots__ = ["path_fn", "save_fn", "load_fn", "exist_fn"]

    def __init__(self, path_fn, save_fn, load_fn, exist_fn=None):
        from os.path import exists
        exist_fn = exist_fn or exists

        assert callable(path_fn)
        assert callable(save_fn)
        assert callable(load_fn)
        assert callable(exist_fn)

        self.path_fn = path_fn
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.exist_fn = exist_fn

    def __getitem__(self, idx):
        path = self.path_fn(idx)
        return self.load_fn(path)

    def __setitem__(self, idx, val):
        path = self.path_fn(idx)
        self.save_fn(path, val)

    def __contains__(self, idx):
        path = self.path_fn(idx)
        return self.exist_fn(path)
