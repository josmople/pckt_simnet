from functools import partial
import model as M
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb

from fewshot import *
from dataloader import *


load = partial(load_iscxvpn2016_bit, pcap_dir="D://Datasets/ISCXVPN2016/", h5_dir="D://Datasets/packets-15k/")

# classifier = M.SimnetClassifier(in_channels=416)
classifier = M.ProtonetClassifier(in_channels=416, out_channels=5)

solver = FewshotSolver(classifier)

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
