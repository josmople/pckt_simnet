import torch
import utils.data as D


class H5TensorCache(D.cache.Cache):
    import h5py
    import torch
    import numpy

    def __init__(self, filepath, ds_data="data", ds_flag="flag"):

        self.db = self.h5py.File(self.prepare_path(filepath), "a")
        self.ds_data = ds_data
        self.ds_flag = ds_flag

    def prepare_path(self, filepath):
        from os import makedirs
        from os.path import dirname
        dirpath = dirname(filepath)
        if len(dirpath) == 0:
            dirpath = "."
        makedirs(dirpath, exist_ok=True)
        return filepath

    @property
    def flag(self) -> h5py.Dataset:
        if self.ds_flag in self.db:
            return self.db[self.ds_flag]
        return None

    @property
    def data(self) -> h5py.Dataset:
        if self.ds_data in self.db:
            return self.db[self.ds_data]
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        if self.data is None:
            raise ValueError("h5py file is not yet initialized")
        if idx >= len(self.data):
            raise IndexError(f"Index ({idx}) is out of range [0,{len(self.data)})")
        if self.flag[idx]:
            tensor = self.data[idx]
            return self.torch.tensor(tensor)
        raise ValueError(f"value at index {idx} is not yet initialized")

    def __setitem__(self, idx: int, val: torch.Tensor):

        if isinstance(val, self.numpy.ndarray):
            val = self.torch.tensor(val)

        dims = val.size()
        if self.data is None:
            self.db.create_dataset(self.ds_data, shape=(idx, *dims), maxshape=(None, *dims), chunks=True)
        if self.flag is None:
            self.db.create_dataset(self.ds_flag, shape=(idx,), dtype=self.numpy.bool, maxshape=(None,), chunks=True)

        if idx >= len(self.data):
            self.data.resize(idx + 1, axis=0)
            self.flag.resize(idx + 1, axis=0)

        self.data[idx] = val.numpy()
        self.flag[idx] = True

    def __contains__(self, idx: int):
        if self.flag is None:
            return False
        if idx > len(self.flag):
            return False
        return bool(self.flag[idx])


if __name__ == "__main__":
    import cache as C
    import torch as t

    d = C.H5TensorCache("D://Datasets/ISCXVPN2016_H5/ICQchat2-pcapng.h5")
    # d[0] = t.ones(3, 5, 2)

    print(d[0])
