from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import model as M
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb
import pytorch_lightning.metrics.classification as plmc

from dataset import FewshotDatasetManager
import typing as _T

import utils as U
from dataloader import *


class FewshotSolver(pl.LightningModule, M.FewshotClassifier):

    def __init__(self, network: M.FewshotClassifier, n_classes=5, lr=1e-4):
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

    def forward(self, queries: torch.Tensor, *supports: _T.List[torch.Tensor]) -> torch.Tensor:
        return self.network(queries, *supports)

    def validation_step(self, batch: _T.List[torch.Tensor], batch_idx: int):
        queries, labels, *supports = batch
        logits = self.network(queries, *supports)

        for category, evaluator in self.evaluators.items():
            self.log(f"metrics/{category}", evaluator(logits, labels))

    def training_step(self, batch: _T.List[torch.Tensor], batch_idx: int):
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


load = partial(load_iscxvpn2016_bit, pcap_dir="D://Datasets/ISCXVPN2016/", h5_dir="D://Datasets/packets-15k/")

# solver = FewshotSolver(M.ProtonetClassifier(in_channels=416, out_channels=5))
solver = FewshotSolver(M.SimnetClassifier(in_channels=416))

datasets = FewshotDatasetManager(
    seen_classes={
        "facebook": load("facebook", "vpn"),
        "youtube": load("youtube", "vpn"),
        "vimeo": load("vimeo", "vpn"),
        "email": load("email", "gmail"),
        "hangouts": load("hangouts", "vpn"),
        "torrent": load("torrent"),
        "icq": load("icq")
    },
    unseen_classes={
        "aim": load("aim", "vpn"),
        "netflix": load("netflix", "vpn"),
        "ftps": load("ftps", "vpn"),
        "sftp": load("sftp", "vpn"),
        "scp": load("scp", "vpn"),
    },
    n_classes=5, n_support=10, n_queries=1000
)
trainer = pl.Trainer(gpus=1, max_epochs=1000, log_every_n_steps=1, precision=16, check_val_every_n_epoch=1, auto_lr_find=True, callbacks=[
    FewshotDatasetReplacement(datasets, every_batch=20),
    plcb.ModelCheckpoint()
])

trainer.tune(solver, train_dataloader=datasets.train_dataloader())
trainer.fit(
    solver,
    train_dataloader=datasets.train_dataloader(),
    val_dataloaders=[
        datasets.val_dataloader(seen=False, unseen=True),
        # datasets.val_dataloader(seen=True, unseen=False),
        # datasets.val_dataloader(seen=False, unseen=True),
    ]
)
