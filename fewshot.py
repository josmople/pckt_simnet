import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import pytorch_lightning.metrics as plmc

import utils.data as data
import typing as T


class FewshotClassifier(nn.Module):

    def __call__(self, queries: torch.Tensor, *supports: T.List[torch.Tensor]) -> torch.Tensor:
        return super().__call__(queries, *supports)

    def forward(self, queries: torch.Tensor, *supports: T.List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class FewshotDatasetManager(pl.LightningDataModule):

    def __init__(
            self,
            seen_classes: T.Dict[str, data.Dataset] = None, unseen_classes: T.Dict[str, data.Dataset] = None,
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
            train_ds, test_ds, val_ds = data.random_split(ds, [train_size, test_size, val_size], generator=self.generator)

            self.datasets_train_seen[f"train:seen:{name}"] = train_ds
            self.datasets_test_seen[f"test:seen:{name}"] = test_ds
            self.datasets_val_seen[f"val:seen:{name}"] = val_ds

        for name, ds in (unseen_classes or {}).items():
            tot = len(ds)
            val_size = int(tot * (unseen_val_ratio or 0))
            test_size = tot - val_size
            test_ds, val_ds = data.random_split(ds, [test_size, val_size], generator=self.generator)

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

    def generate_sequence(self, datasets: T.Dict[str, data.Dataset], n_classes: int = None, generator: torch.Generator = None):
        n_classes = n_classes or self.n_classes
        generator = generator or self.generator

        assert len(datasets) >= n_classes

        from itertools import combinations
        from random import shuffle

        choices = list(combinations(datasets.keys(), n_classes))
        access = torch.randperm(len(choices), generator=generator)

        idx = 0
        while True:
            try:
                choice = choices[access[idx]]
                idx += 1
                yield {name: datasets[name] for name in choice}
            except IndexError:
                idx = 0
                access = torch.randperm(len(choices), generator=generator)

    def select_batch(self, dataset: data.Dataset, size: int, generator: torch.Generator = None):
        generator = generator or self.generator
        dataloader = data.DataLoader(dataset, batch_size=size, shuffle=True, generator=generator)
        return next(iter(dataloader))

    def build_dataloader(self, datasets: T.Dict[str, data.Dataset], n_support: int = None, n_queries: int = None, generator: torch.Generator = None):
        n_support = n_support or self.n_support
        n_queries = n_queries or self.n_queries
        generator = generator or self.generator

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
            support = self.select_batch(dataset, size=n_support, generator=generator)
            support = data.dconst(support, len(queries))
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

        dataset = data.dzip(queries, labels, *supports)
        return data.DataLoader(dataset, batch_size=self.n_queries, collate_fn=collate_fn, shuffle=True, generator=generator)

    def train_dataloader(self, n_support=None, n_queries=None) -> data.DataLoader:
        datasets = next(self.sequence_train)
        assert datasets is not None
        return self.build_dataloader(datasets, n_support=n_support or self.n_support, n_queries=n_queries or self.n_queries)

    def test_dataloader(self, n_support=None, n_queries=None, seen=True, unseen=True) -> data.DataLoader:
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

    def val_dataloader(self, n_support=None, n_queries=None, seen=True, unseen=True) -> data.DataLoader:
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


class FewshotSolver(pl.LightningModule, FewshotClassifier):

    def __init__(self, network: FewshotClassifier, n_classes=5, lr=1e-4):
        pl.LightningModule.__init__(self)

        self.network = network
        self.lr = lr
        self.evaluators = nn.ModuleDict({
            "accuracy": plmc.Accuracy(),
            "precision": plmc.Precision(num_classes=n_classes),
            "recall": plmc.Recall(num_classes=n_classes),
            "fbeta": plmc.FBeta(num_classes=n_classes),
            "f1": plmc.F1(num_classes=n_classes)
        })

    def forward(self, queries: torch.Tensor, *supports: T.List[torch.Tensor]) -> torch.Tensor:
        return self.network(queries, *supports)

    def validation_step(self, batch: T.List[torch.Tensor], batch_idx: int, dataloader_idx: int):
        queries, labels, *supports = batch
        logits = self.network(queries, *supports)

        for category, evaluator in self.evaluators.items():
            self.log(f"metrics/{dataloader_idx}/{category}", evaluator(logits, labels))

    def training_step(self, batch: T.List[torch.Tensor], batch_idx: int):
        queries, labels, *supports = batch
        logits = self.network(queries, *supports)
        class_loss = F.cross_entropy(logits, labels)
        self.log("losses/class_loss", class_loss, on_step=True)
        return class_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class FewshotDatasetReplacement(pl.Callback):

    def __init__(self, datamodule: FewshotDatasetManager, every_batch=10):
        self.datamodule = datamodule
        self.every_batch = every_batch
        self._count = 0

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: FewshotSolver, *args):
        if self._count % self.every_batch == 0:
            trainer.train_dataloader = self.datamodule.train_dataloader()
            print("Classes: ", list(self.datamodule.last_datasets.keys()))
        self._count += 1
