
#####################################################
# Basic Functions
#####################################################

def identity_transform(x):
    return x


#####################################################
# Basic Functions
#####################################################

def is_dataset_like(ds):
    if callable(getattr(ds, "__getitem__", None)) and callable(getattr(ds, "__len__", None)):
        return True
    return False


def is_iterdataset_like(ds, strict=False):
    if callable(getattr(ds, "__iter__", None)):
        return True
    if not strict and is_dataset_like(ds):
        return True
    return False


#####################################################
# Dataset Operations
#####################################################

def dpipe(dataset=None, operators=[]):
    if dataset is None:
        from functools import partial
        return partial(dpipe, operators=operators)

    if callable(operators):
        return operators(dataset)

    for operator in operators:
        dataset = operator(dataset)
    return dataset


def dconst(value, length, is_fn=None):
    from .dataset import ConstantDataset
    if is_fn is None:
        is_fn = callable(value)
    return ConstantDataset(value, length, is_fn)


def dmap(values=None, transform=None, as_iter=False):
    if values is None:
        from functools import partial
        return partial(dmap, transform=transform, force_iter=as_iter)

    transform = transform or identity_transform
    if isinstance(transform, (list, tuple)):
        from torchvision.transforms import Compose
        transform = Compose(transform)

    if is_dataset_like(values) and not as_iter:
        from .dataset import ValueDataset
        return ValueDataset(values, transform)

    if is_iterdataset_like(values):
        from .dataset import ValueIterableDataset
        return ValueIterableDataset(values, transform)

    raise Exception("Parameter datasets must contain element that implement Dataset or IterableDataset")


def dcat(*datasets, as_iter=False):
    if all([is_dataset_like(ds) for ds in datasets]) and not as_iter:
        from .dataset import Dataset, ConcatDataset
        copy = list(datasets)
        for idx, ds in enumerate(datasets):
            if not isinstance(ds, Dataset):
                copy[idx] = dmap(ds, as_iter=False)
        return ConcatDataset(copy)

    if all([is_iterdataset_like(ds) for ds in datasets]):
        from .dataset import IterableDataset, ChainDataset
        copy = list(datasets)
        for idx, ds in enumerate(datasets):
            if not isinstance(ds, IterableDataset):
                copy[idx] = dmap(ds, as_iter=True)
        return ChainDataset(copy)

    raise Exception("Parameter datasets must contain element that implement Dataset or IterableDataset")


def dzip(*datasets, zip_transform=None, as_iter=False):
    if len(datasets) <= 0:
        from functools import partial
        return partial(dzip, zip_transform=zip_transform)

    zip_transform = zip_transform or identity_transform
    if isinstance(zip_transform, (list, tuple)):
        from torchvision.transforms import Compose
        zip_transform = Compose(zip_transform)

    if all([is_dataset_like(ds) for ds in datasets]) and not as_iter:
        from .dataset import ZipDataset
        return ZipDataset(datasets, zip_transform)

    if all([is_iterdataset_like(ds) for ds in datasets]):
        from .dataset import ZipIterableDataset
        return ZipIterableDataset(datasets, zip_transform)

    raise Exception("Parameter datasets must contain element that implement Dataset or IterableDataset")


def dproduct(*datasets, comb_transform=None, custom_indexer=None, as_iter=False):
    if len(datasets) <= 0:
        from functools import partial
        return partial(dproduct, comb_transform=comb_transform)

    comb_transform = comb_transform or identity_transform
    if isinstance(comb_transform, (list, tuple)):
        from torchvision.transforms import Compose
        comb_transform = Compose(comb_transform)

    if all([is_dataset_like(ds) for ds in datasets]) and not as_iter:
        from .dataset import ProductDataset
        return ProductDataset(datasets, comb_transform, indexer=custom_indexer)

    if all([is_iterdataset_like(ds) for ds in datasets]):
        from .dataset import ProductIterableDataset
        return ProductIterableDataset(datasets, comb_transform, indexer=custom_indexer)

    raise Exception("Parameter datasets must contain element that implement Dataset or IterableDataset")


def daugment(dataset=None, aug_fn=None):
    if dataset is None:
        from functools import partial
        return partial(daugment, aug_fn=aug_fn)

    aug_fn = aug_fn or identity_transform
    if isinstance(aug_fn, (list, tuple)):
        from torchvision.transforms import Compose
        aug_fn = Compose(aug_fn)

    from .dataset import AugmentedDataset
    return AugmentedDataset(dataset, aug_fn)


def dcache(dataset=None, cache=None, enable=True):
    assert cache is not None

    if dataset is None:
        from functools import partial
        return partial(dcache, cache=cache, enable=enable)

    if enable:
        not_cache = any([
            getattr(cache, "__getitem__", None) is None,
            getattr(cache, "__setitem__", None) is None,
            getattr(cache, "__contains__", None) is None
        ])
        if callable(cache) and not_cache:
            cache = cache()

        from .dataset import CachedDataset
        return CachedDataset(dataset, cache)

    return dataset


def dlazy(dataset_fn=None, dummy_len=None, as_iter=False):
    if dataset_fn is None:
        from functools import partial
        return partial(dataset_fn, dummy_len=dummy_len, as_iter=as_iter)

    if as_iter:
        from .dataset import LazyIterableDataset
        return LazyIterableDataset(dataset_fn)

    from .dataset import LazyDataset
    return LazyDataset(dataset_fn, dummy_len)


#####################################################
# Dataset Constructors
#####################################################

def numbers(size, transform=None):
    return dmap(range(size), transform)


def glob_files(paths, transform=None, recursive=False, unique=True, sort=True, sort_key=None, sort_reverse=False):
    from .utils import glob
    return dmap(glob(paths, recursive=recursive, unique=unique, sort=sort, sort_key=sort_key, sort_reverse=sort_reverse), transform)


def index_files(pathquery, transform=None, maxsize=None):
    from os import walk
    from os.path import dirname

    def generate_path(idx):
        return pathquery.format(idx, idx=idx, index=idx)

    if maxsize is None:
        dirpath = dirname(generate_path(0))
        maxsize = len(next(walk(dirpath))[2])

    return numbers(maxsize, [
        generate_path,
        transform or identity_transform
    ])


def images(paths, transform=None, img_loader="pil", img_autoclose=True):
    if transform is None:
        img_autoclose = False
        transform = identity_transform

    if isinstance(transform, (list, tuple)):
        assert all([callable(t) for t in transform])
        from torchvision.transforms import Compose
        transform = Compose(transform)
    assert callable(transform)

    if isinstance(img_loader, str):
        from importlib import import_module
        img_loader = img_loader.lower()

        if img_loader == "pil":
            img_loader = import_module("PIL.Image").open
        elif img_loader == "cv2":
            img_loader = import_module("cv2").imread
        elif img_loader == "imageio":
            get_reader = import_module("imageio").get_reader

            def imageio_loader(path):
                return get_reader(path).get_next_data()
            img_loader = imageio_loader

    if not callable(img_loader):
        from importlib.util import find_spec as module_exists
        if module_exists("PIL"):
            from PIL.Image import open as pil_loader
            img_loader = pil_loader
        else:
            from .utils import eprint
            eprint(
                "Default image loader is pillow (PIL).",
                "Module 'PIL' not found!",
                "Try 'pip install imageio' or 'pip install pillow' or provide custom 'img_loader'"
            )

    if img_autoclose:
        def img_transform(path):
            img = img_loader(path)
            out = transform(img)
            if callable(getattr(img, "close", None)):
                if img == out:
                    from .utils import eprint
                    eprint(f"Warning: Auto-closing image but image is unprocessed: {path}")
                img.close()
            return out
    else:
        def img_transform(path):
            return transform(img_loader(path))

    return dmap(paths, img_transform)


def glob_images(paths, transform=None, img_loader="pil", img_autoclose=True, glob_recursive=False, glob_unique=True, glob_sort=True, sort_key=None, sort_reverse=False):
    paths = glob_files(paths, recursive=glob_recursive, unique=glob_unique, sort=glob_sort, sort_key=sort_key, sort_reverse=sort_reverse)
    return images(paths, transform, img_loader=img_loader, img_autoclose=img_autoclose)


def index_images(pathquery, transform, img_loader="pil", img_autoclose=True, maxsize=None):
    paths = index_files(pathquery, maxsize=maxsize)
    return images(paths, transform, img_loader=img_loader, img_autoclose=img_autoclose)


def tensors(paths, transform=None, tensor_loader=None):
    transform = transform or identity_transform
    if isinstance(transform, (list, tuple)):
        from torchvision.transforms import Compose
        transform = Compose(transform)
    assert callable(transform)

    if not callable(tensor_loader):
        try:
            from torch import load as torch_loader
            tensor_loader = torch_loader
        except ModuleNotFoundError as e:
            from .utils import eprint
            eprint("Default tensor loader is PyTorch (torch). Module 'torch' not found! Install PyTorch or provide custom 'tensor_loader'")
            raise e

    def tensor_transform(path):
        tensor = torch_loader(path)
        return transform(tensor)

    return dmap(paths, transform=tensor_transform)


def glob_tensor(paths, transform=None, tensor_loader=None, glob_recursive=False, glob_unique=True, glob_sort=True, sort_key=None, sort_reverse=False):
    paths = glob_files(paths, recursive=glob_recursive, unique=glob_unique, sort=glob_sort, sort_key=sort_key, sort_reverse=sort_reverse)
    return tensors(paths, transform, tensor_loader=tensor_loader)


def index_tensor(pathquery, transform, tensor_loader=None, maxsize=None):
    paths = index_files(pathquery, maxsize=maxsize)
    return tensors(paths, transform, tensor_loader=tensor_loader)


#####################################################
# Cache Constructors
#####################################################

def cache_create(load_fn, save_fn, exist_fn):
    from .cache import LambdaCache
    return LambdaCache(save_fn=save_fn, load_fn=load_fn, exist_fn=exist_fn)


def cache_dict(preloaded_data=None):
    from .cache import DictCache
    return DictCache(preloaded_data)


def cache_file(cache_dir, load_fn, save_fn, make_dir=True):

    path_fn = None
    error_msg = "cached_dir must be a string or a callable"

    if isinstance(cache_dir, str):
        def path_fn(idx):
            return cache_dir.format(idx=idx)
        error_msg = "The parameter 'cache_dir:str' must contain the token '{idx}' (e.g. 'cache/{idx:04}.pt') for string formatting"

    elif callable(cache_dir):
        path_fn = cache_dir
        error_msg = "The parameter 'cache_dir:Callable' must receive one argument of type 'int' and return value of type 'str'"

    try:
        sample_filepath = path_fn(0)
        assert isinstance(sample_filepath, str)
    except Exception as e:
        from .utils import eprint
        eprint(error_msg)
        raise e

    if make_dir:
        from os.path import dirname
        dirpath = dirname(sample_filepath)

        from os import makedirs
        makedirs(dirpath, exist_ok=True)

    from .cache import FileCache
    cache = FileCache(path_fn=path_fn, save_fn=save_fn, load_fn=load_fn)

    return cache


def cache_tensor(cache_dir, make_dir=True):
    from functools import wraps
    from torch import load, save

    @wraps(load)
    def load_pytorch_tensor(path):
        return load(path)

    @wraps(save)
    def save_pytorch_tensor(path, tensor):
        return save(tensor, path)

    return cache_file(cache_dir, load_fn=load_pytorch_tensor, save_fn=save_pytorch_tensor, make_dir=make_dir)


def cache_text(cache_dir, as_array=False, make_dir=True):
    from os import linesep

    if as_array:
        def load_text(path):
            with open(path, "r") as f:
                lines = f.readlines()
            return list(filter(lambda x: x.strip(linesep), lines))

        def save_text(path, lines):
            text = str.join(linesep, lines)
            with open(path, "w+") as f:
                f.write(text)
    else:
        def load_text(path):
            with open(path, "r") as f:
                return f.read()

        def save_text(path, text):
            with open(path, "w+") as f:
                f.write(text)

    return cache_file(cache_dir, load_fn=load_text, save_fn=save_text, make_dir=make_dir)


def cache_json(cache_dir, load_kwds=None, save_kwds=None, make_dir=True):
    from json import load, dump

    def load_json(path):
        with open(path, "r") as f:
            return load(f, **(load_kwds or {}))

    def save_json(path, obj):
        with open(path, "w+") as f:
            return dump(obj, **(save_kwds or {}))

    return cache_file(cache_dir, load_fn=load_json, save_fn=save_json, make_dir=make_dir)


#####################################################
# Cache Dataset Macros
#####################################################

def dcache_dict(dataset, preloaded_data=None, enable=True):
    from functools import partial
    cache_gen = partial(cache_dict, preloaded_data=preloaded_data)
    return dcache(dataset, cache_gen, enable)


def dcache_file(dataset, cache_dir, load_fn, save_fn, make_dir=True, enable=True):
    from functools import partial
    cache_gen = partial(cache_file, cache_dir=cache_dir, load_fn=load_fn, save_fn=save_fn, make_dir=make_dir)
    return dcache(dataset, cache_gen, enable)


def dcache_tensor(dataset, cache_dir, make_dir=True, enable=True):
    from functools import partial
    cache_gen = partial(cache_tensor, cache_dir=cache_dir, make_dir=make_dir)
    return dcache(dataset, cache_gen, enable)


def dcache_text(dataset, cache_dir, as_array=False, make_dir=True, enable=True):
    from functools import partial
    cache_gen = partial(cache_text, cache_dir=cache_dir, array=as_array, make_dir=make_dir)
    return dcache(dataset, cache_gen, enable)


def dcache_json(dataset, cache_dir, load_kwds=None, save_kwds=None, make_dir=True, enable=True):
    from functools import partial
    cache_gen = partial(cache_json, cache_dir=cache_dir, load_kwds=load_kwds, save_kwds=save_kwds, make_dir=make_dir)
    return dcache(dataset, cache_gen, enable)
