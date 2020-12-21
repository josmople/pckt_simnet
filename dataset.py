import utils as U
import typing as T

import torch
import pytorch_lightning as pl


class PcapH5Dataset(U.data.ValueDataset):

    def __init__(self, pcap_path, transform=None, h5db_path=None, h5ds_name="all", max_packets=15000):
        self.pcap_path = pcap_path
        self.transform = transform
        self.h5db_path = h5db_path
        self.h5ds_name = h5ds_name
        self.max_packets = max_packets

        self.h5db = None
        super().__init__(self.load_data(), transform)

    def reload(self):
        self.close()
        self.values = self.load_data()

    def load_data(self):
        import os
        import os.path as path

        import scapy.all as sp
        import h5py

        import numpy as np

        from tqdm import tqdm

        if self.h5db_path is None:
            with sp.PcapReader(self.pcap_path) as reader:
                return reader.read_all(self.max_packets)

        if path.exists(self.h5db_path):
            self.h5db = h5py.File(self.h5db_path, "r")
            return self.h5db[self.h5ds_name]

        dirpath = path.dirname(self.h5db_path)
        if len(dirpath) == 0:
            dirpath = "."
        os.makedirs(dirpath, exist_ok=True)

        with h5py.File(self.h5db_path, "w") as db, sp.PcapReader(self.pcap_path) as packets:
            cache = None

            for idx, packet in tqdm(enumerate(packets)):

                if self.max_packets and idx >= self.max_packets:
                    break

                packet.getlayer(0).dst = '0'
                packet.getlayer(0).src = '0'

                if packet.getlayer(1).name == 'IPv6':
                    packet.getlayer(1).src = "0000::0000:0000:0000:0000"
                    packet.getlayer(1).dst = '0000::00'
                else:
                    packet.getlayer(1).src = "0"
                    packet.getlayer(1).dst = '0'

                if packet.getlayer(2) is not None:
                    packet.getlayer(2).sport = 0
                    packet.getlayer(2).dport = 0
                    hex_bytes = sp.raw(packet)

                    if len(hex_bytes) >= 52:
                        num_list = [int(n) for n in hex_bytes][:40 + 12]
                    else:
                        missing = 52 - len(hex_bytes)
                        padding = [0] * missing
                        num_list = [int(n) for n in hex_bytes] + padding
                    num_list = np.asarray(num_list).astype(int).reshape(-1)

                    if cache is None:
                        dims = num_list.shape
                        cache = db.create_dataset(self.h5ds_name, shape=(1, *dims), maxshape=(None, *dims), chunks=True)

                    if idx >= len(cache):
                        cache.resize(idx + 1, axis=0)
                    cache[idx] = num_list

        self.h5db = h5py.File(self.h5db_path, "r")
        return self.h5db[self.h5ds_name]

    def close(self):
        if self.h5db:
            self.h5db.close()
        self.h5db = None
        self.values = []


class FewshotDatasetManager(pl.LightningDataModule):

    def __init__(
            self,
            seen_classes: T.Dict[str, U.data.Dataset] = None, unseen_classes: T.Dict[str, U.data.Dataset] = None,
            n_classes=5, n_support=1, n_queries=1000,
            seen_test_ratio=0.2, seen_val_ratio=0.1, unseen_val_ratio=0.1,
            generator: torch.Generator = None
    ):

        self.n_classes = n_classes
        self.n_support = n_support
        self.n_queries = n_queries

        self.generator = generator

        seen_classes = seen_classes or {}
        unseen_classes = unseen_classes or {}

        self.datasets_train_seen = {}

        self.datasets_test_seen = {}
        self.datasets_test_unseen = {}

        self.datasets_val_seen = {}
        self.datasets_val_unseen = {}

        for name, ds in (seen_classes or {}).items():
            tot = len(ds)
            val_size = int(tot * (seen_val_ratio or 0))
            test_size = int(tot * (seen_test_ratio or 0))
            train_size = tot - val_size - test_size
            train_ds, test_ds, val_ds = U.data.random_split(ds, [train_size, test_size, val_size], generator=self.generator)

            self.datasets_train_seen[f"train:seen:{name}"] = train_ds
            self.datasets_test_seen[f"test:seen:{name}"] = test_ds
            self.datasets_val_seen[f"val:seen:{name}"] = val_ds

        for name, ds in (unseen_classes or {}).items():
            tot = len(ds)
            val_size = int(tot * (unseen_val_ratio or 0))
            test_size = tot - val_size
            test_ds, val_ds = U.data.random_split(ds, [test_size, val_size], generator=self.generator)

            self.datasets_test_unseen[f"test:unseen:{name}"] = test_ds
            self.datasets_val_unseen[f"val:unseen:{name}"] = val_ds

        if len(self.datasets_train_seen) > 0:
            self.sequence_train = self.generate_sequence(self.datasets_train_seen, n_classes)

        if len(self.datasets_test_seen) > 0:
            self.sequence_test_seen = self.generate_sequence(self.datasets_test_seen, n_classes)
        if len(self.datasets_test_unseen) > 0:
            self.sequence_test_unseen = self.generate_sequence(self.datasets_test_unseen, n_classes)
        if len(self.datasets_test_seen) > 0:
            self.sequence_test = self.generate_sequence({**self.datasets_test_seen, **self.datasets_test_unseen}, n_classes)

        if len(self.datasets_val_seen) > 0:
            self.sequence_val_seen = self.generate_sequence(self.datasets_val_seen, n_classes)
        if len(self.datasets_val_unseen) > 0:
            self.sequence_val_unseen = self.generate_sequence(self.datasets_val_unseen, n_classes)
        if len(self.datasets_val_seen) > 0:
            self.sequence_val = self.generate_sequence({**self.datasets_val_seen, **self.datasets_val_unseen}, n_classes)

    def generate_sequence(self, datasets: T.Dict[str, U.data.Dataset], n_classes=None):
        n_classes = n_classes or self.n_classes

        assert len(datasets) >= n_classes

        from itertools import combinations
        from random import shuffle

        choices = list(combinations(datasets.keys(), n_classes))
        shuffle(choices)

        idx = 0
        while True:
            try:
                choice = choices[idx]
                idx += 1
                yield {name: datasets[name] for name in choice}
            except IndexError:
                idx = 0
                choices = list(combinations(datasets.keys(), n_classes))
                shuffle(choices)

    def select_batch(self, dataset, n, generator=None):
        support = U.data.DataLoader(dataset, batch_size=n, shuffle=True, generator=generator or self.generator)
        support = next(iter(support))

    def build_dataloader(self, datasets: T.Dict[str, U.data.Dataset], n_support=None, n_queries=None, generator=None):
        n_support = n_support or self.n_support
        n_queries = n_queries or self.n_queries

        self.last_datasets = datasets
        dslist = list(datasets.values())

        queries = dslist[0]
        labels = [0] * len(dslist[0])

        for i in range(1, len(dslist)):
            queries += dslist[i]
            labels += [i] * len(dslist[i])

        # Select supports
        # Will be constant for all queries
        supports = []
        for dataset in dslist:
            support = U.data.DataLoader(dataset, batch_size=n_support, shuffle=True)
            support = next(iter(support))
            support = U.data.dconst(support, len(queries))
            supports.append(support)

        def collate_fn(batch):
            queries = []
            labels = []
            support = []  # Don't touch support since its constant through all iterations
            for row in batch:
                query, label, *support = row
                queries.append(query)
                labels.append(label)
            queries = torch.stack(queries, dim=0)
            labels = torch.tensor(labels, device=queries.device)
            return queries, labels, *support

        dataset = U.data.dzip(queries, labels, *supports)
        return U.data.DataLoader(dataset, batch_size=self.n_queries, collate_fn=collate_fn, shuffle=True)

    def train_dataloader(self, n_support=None, n_queries=None) -> U.data.DataLoader:
        datasets = next(self.sequence_train)
        assert datasets is not None
        return self.build_dataloader(datasets, n_support=n_support or self.n_support, n_queries=n_queries or self.n_queries)

    def test_dataloader(self, n_support=None, n_queries=None, seen=True, unseen=True) -> U.data.DataLoader:
        assert seen or unseen, "At least one should be True"
        datasets = None
        if seen and unseen:
            datasets = next(self.sequence_test)
        elif seen:
            datasets = next(self.sequence_test_seen)
        elif unseen:
            datasets = next(self.sequence_test_unseen)
        assert datasets is not None
        return self.build_dataloader(datasets, n_support=n_support or self.n_support, n_queries=n_queries or self.n_queries)

    def val_dataloader(self, n_support=None, n_queries=None, seen=True, unseen=True) -> U.data.DataLoader:
        assert seen or unseen, "At least one should be True"
        datasets = None
        if seen and unseen:
            datasets = next(self.sequence_val)
        elif seen:
            datasets = next(self.sequence_val_seen)
        elif unseen:
            datasets = next(self.sequence_val_unseen)
        assert datasets is not None
        return self.build_dataloader(datasets, n_support=n_support or self.n_support, n_queries=n_queries or self.n_queries)
