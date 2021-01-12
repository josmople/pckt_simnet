from torch.utils.data.dataset import *


class ValueDataset(Dataset):

    __slots__ = ["values", "transform"]

    def __init__(self, values, transform=None):
        assert callable(getattr(values, "__len__", None))
        assert callable(getattr(values, "__getitem__", None))
        assert callable(transform) or transform is None

        self.values = values
        self.transform = transform

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        value = self.values[idx]
        if self.transform is None:
            return value
        return self.transform(value)


class ValueIterableDataset(IterableDataset):

    __slots__ = ["values", "transform"]

    def __init__(self, values, transform=None):
        from collections import Iterable
        assert isinstance(values, Iterable)
        assert callable(transform) or transform is None

        self.values = values
        self.transform = transform

    @staticmethod
    def generator(values, transform):
        if transform is None:
            transform = (lambda x: x)

        for v in values:
            yield transform(v)

    def __iter__(self):
        return self.generator(self.values, self.transform)


class ZipDataset(Dataset):

    __slots__ = ["datasets", "zip_transform"]

    def __init__(self, datasets, zip_transform=None):
        assert len(datasets) > 0

        assert all([callable(getattr(ds, "__len__", None)) for ds in datasets])
        assert all([callable(getattr(ds, "__getitem__", None)) for ds in datasets])

        assert callable(zip_transform) or zip_transform is None

        self.datasets = datasets
        self.zip_transform = zip_transform

    @property
    def sizes(self):
        return tuple([len(ds) for ds in self.datasets])

    def __len__(self):
        return min(self.sizes)

    def __getitem__(self, idx):
        array = []
        for ds in self.datasets:
            array.append(ds[idx])
        if self.zip_transform is None:
            return tuple(array)
        return self.zip_transform(tuple(array))


class ZipIterableDataset(IterableDataset):

    __slots__ = ["datasets", "zip_transform"]

    def __init__(self, datasets, zip_transform=None):
        assert len(datasets) > 0

        from collections import Iterable
        assert all([isinstance(ds, Iterable) for ds in datasets])

        assert callable(zip_transform) or zip_transform is None

        self.datasets = datasets
        self.zip_transform = zip_transform

    @staticmethod
    def generator(datasets, zip_transform):
        if zip_transform is None:
            zip_transform = (lambda x: x)

        for vals in zip(*datasets):
            yield zip_transform(vals)

    def __iter__(self):
        return self.generator(self.datasets, self.zip_transform)


class ProductDataset(Dataset):

    __slots__ = ["datasets", "comb_transform"]

    def __init__(self, datasets, comb_transform=None, indexer=None):
        assert len(datasets) > 0

        assert all([callable(getattr(ds, "__len__", None)) for ds in datasets])
        assert all([callable(getattr(ds, "__getitem__", None)) for ds in datasets])

        assert callable(comb_transform) or comb_transform is None

        self.datasets = datasets
        self.comb_transform = comb_transform

        if callable(indexer):
            self.indexer = indexer

    @staticmethod
    def indexer(i, sizes):
        prod = 1
        for s in sizes:
            prod *= s

        out = []
        for s in sizes:
            prod //= s
            q = i // prod
            i = i % prod
            out.append(q)

        return tuple(out)

    @property
    def sizes(self):
        return tuple([len(ds) for ds in self.datasets])

    def __len__(self):
        from functools import reduce
        from operator import mul
        return reduce(mul, self.sizes, 1)

    def __getitem__(self, idx):
        idxs = self.indexer(idx, self.sizes)
        array = []
        for i, ds in zip(idxs, self.datasets):
            array.append(ds[i])

        if self.comb_transform is None:
            return tuple(array)
        return self.comb_transform(tuple(array))


class ProductIterableDataset(IterableDataset):

    __slots__ = ["datasets", "comb_transform"]

    def __init__(self, datasets, comb_transform, indexer=None):
        assert len(datasets) > 0

        from collections import Iterable
        assert all([isinstance(ds, Iterable) for ds in datasets])

        assert callable(comb_transform) or comb_transform is None

        self.datasets = datasets
        self.comb_transform = comb_transform

        if callable(indexer):
            self.indexer = indexer

    @staticmethod
    def indexer(i, sizes):
        prod = 1
        for s in sizes:
            prod *= s

        out = []
        for s in sizes:
            prod //= s
            q = i // prod
            i = i % prod
            out.append(q)

        return tuple(out)

    @staticmethod
    def generator(datasets, comb_transform):
        if comb_transform is None:
            comb_transform = (lambda x: x)

        from itertools import product
        for vals in product(*datasets):
            yield comb_transform(vals)

    def __iter__(self):
        return self.generator(self.datasets, self.comb_transform)


class AugmentedDataset(IterableDataset):

    __slots__ = ["values", "augment"]

    def __init__(self, values, augment):
        from collections import Iterable
        assert isinstance(values, Iterable)
        assert callable(augment) or augment is None

        self.values = values
        self.augment = augment

    @staticmethod
    def generator(values, augment):
        if augment is None:
            augment = (lambda x: [x])

        for v in values:
            for o in augment(v):
                yield o

    def __iter__(self):
        return self.generator(self.values, self.augment)


class CachedDataset(Dataset):

    __slots__ = ["dataset", "cache"]

    def __init__(self, dataset, cache):
        assert callable(getattr(dataset, "__len__", None))
        assert callable(getattr(dataset, "__getitem__", None))

        assert callable(getattr(cache, "__getitem__", None))
        assert callable(getattr(cache, "__setitem__", None))
        assert callable(getattr(cache, "__contains__", None))

        self.dataset = dataset
        self.cache = cache

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]


class LazyDataset(Dataset):

    __slots__ = ["dataset", "dataset_fn", "dummy_len"]

    def __init__(self, dataset_fn, dummy_len=None):
        self.dataset = None
        self.dataset_fn = dataset_fn
        self.dummy_len = dummy_len

    def _load_dataset(self):
        dataset = self.dataset_fn() if callable(self.dataset_fn) else self.dataset_fn
        assert callable(getattr(dataset, "__len__", None))
        assert callable(getattr(dataset, "__getitem__", None))
        self.dataset = dataset

    def __getitem__(self, idx):
        if self.dataset is None:
            self._load_dataset()
        return self.dataset[idx]

    def __len__(self):
        if self.dataset is None:
            if self.dummy_len is not None:
                dummy_len = self.dummy_len() if callable(self.dummy_len) else self.dummy_len
                return int(dummy_len)
            self._load_dataset()
        return len(self.dataset)


class LazyIterableDataset(IterableDataset):

    __slots__ = ["dataset", "dataset_fn"]

    def __init__(self, dataset_fn):
        self.dataset = None
        self.dataset_fn = dataset_fn

    def _load_dataset(self):
        dataset = self.dataset_fn() if callable(self.dataset_fn) else self.dataset_fn
        from collections import Iterable
        assert isinstance(dataset, Iterable)
        self.dataset = dataset

    def __iter__(self):
        if self.dataset is None:
            self._load_dataset()
        return iter(self.dataset)


class ConstantDataset(Dataset):

    def __init__(self, value, length, is_fn=False):
        self.value = value
        self.length = length
        self.is_fn = is_fn

        if self.is_fn:
            assert callable(self.value)

    def __getitem__(self, idx):
        if idx >= self.length or idx < 0:
            raise IndexError(f"Index must in [0, {self.length}) but got {idx}")
        return self.value() if self.is_fn else self.value

    def __len__(self):
        return self.length
