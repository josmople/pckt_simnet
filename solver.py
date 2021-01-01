import model as M
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb
import pytorch_lightning.trainer as pltr
import pytorch_lightning.loggers as pllog

from fewshot import *
from dataloader import *


iscxvpn2016_loader = iscxvpn2016(pcap_dir="D://Datasets/ISCXVPN2016/", h5_dir="D://Datasets/packets-50k/", max_packets_on_cache=50000, as_bit=True, verbose=True)
ustctfc2016_loader = ustctfc2016(pcap_dir="D://Datasets/USTC-TFC2016/", h5_dir="D://Datasets/USTC-TFC2016-packets-50k/", max_packets_on_cache=50000, as_bit=True, verbose=True)

classifiers = {
    "protonet_1": M.ProtonetClassifier(in_channels=416, mid_channels=[], out_channels=32),
    "protonet_2": M.ProtonetClassifier(in_channels=416, mid_channels=[64], out_channels=32),
    "protonet_3": M.ProtonetClassifier(in_channels=416, mid_channels=[128, 64], out_channels=32),
    "protonet_4": M.ProtonetClassifier(in_channels=416, mid_channels=[256, 128, 64], out_channels=32),
    "protonet_bottleneck_end": M.ProtonetClassifier(in_channels=416, mid_channels=[256, 128, 64], out_channels=10),
    "protonet_bottleneck_mid": M.ProtonetClassifier(in_channels=416, mid_channels=[128, 32, 128], out_channels=32),

    "simnet_simple": M.SimnetClassifier(in_channels=416, channels=[10]),
    "simnet_1": M.SimnetClassifier(in_channels=416, channels=[32]),
    "simnet_2": M.SimnetClassifier(in_channels=416, channels=[64, 32]),
    "simnet_3": M.SimnetClassifier(in_channels=416, channels=[128, 64, 32]),

    "reg_protonet_1": M.ProtonetClassifier(in_channels=416, mid_channels=[], out_channels=32),
    "reg_protonet_2": M.ProtonetClassifier(in_channels=416, mid_channels=[64], out_channels=32),
    "reg_protonet_3": M.ProtonetClassifier(in_channels=416, mid_channels=[128, 64], out_channels=32),
    "reg_protonet_4": M.ProtonetClassifier(in_channels=416, mid_channels=[256, 128, 64], out_channels=32),
    "reg_protonet_bottleneck_end": M.ProtonetClassifier(in_channels=416, mid_channels=[256, 128, 64], out_channels=10),
    "reg_protonet_bottleneck_mid": M.ProtonetClassifier(in_channels=416, mid_channels=[128, 32, 128], out_channels=32),

    "reg_simnet_simple": M.SimnetClassifier(in_channels=416, channels=[10]),
    "reg_simnet_1": M.SimnetClassifier(in_channels=416, channels=[32]),
    "reg_simnet_2": M.SimnetClassifier(in_channels=416, channels=[64, 32]),
    "reg_simnet_3": M.SimnetClassifier(in_channels=416, channels=[128, 64, 32]),
}

for classifier_name, classifier in classifiers.items():
    pl.seed_everything(seed=2020)

    print("----------------------------------------------------------------------------------")
    print(classifier_name)

    datasets = FewshotDatasetManager(
        seen_classes={
            "aim": iscxvpn2016_loader("aim"),
            "email": iscxvpn2016_loader("email", "gmail"),
            "facebook": iscxvpn2016_loader("facebook"),
            "ftps": iscxvpn2016_loader("ftps"),
            "gmail": iscxvpn2016_loader("gmail"),
            "hangouts": iscxvpn2016_loader("hangouts"),
            "icq": iscxvpn2016_loader("icq"),
            "youtube": iscxvpn2016_loader("youtube"),
            "netflix": iscxvpn2016_loader("netflix"),
            "scp": iscxvpn2016_loader("scp"),
            "sftp": iscxvpn2016_loader("sftp"),
            "skype": iscxvpn2016_loader("skype"),
            "spotify": iscxvpn2016_loader("spotify"),
            "vimeo": iscxvpn2016_loader("vimeo"),
            "torrent": iscxvpn2016_loader("torrent"),
        },
        unseen_classes={k: ustctfc2016_loader(k) for k in ustctfc2016_loader.metadata.names()},
        n_classes=5, n_support=10, n_queries=1000
    )

    if classifier_name.startswith("reg"):
        solver = FewshotSolver(classifier, weight_decay=1e-4)
    else:
        solver = FewshotSolver(classifier, weight_decay=0)

    tb_logger = pllog.TensorBoardLogger("logs_bigdata_regularized/" + classifier_name)
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=1,
        max_epochs=30,
        log_every_n_steps=1,
        precision=32,
        check_val_every_n_epoch=1,
        auto_lr_find=True,
        callbacks=[
            FewshotDatasetReplacement(datasets, every_batch=20),
            plcb.ModelCheckpoint()
        ])

    trainer.tune(solver, train_dataloader=datasets.train_dataloader())
    trainer.fit(
        solver,
        train_dataloader=datasets.train_dataloader(),
        val_dataloaders=[
            datasets.val_dataloader(seen=False, unseen=True),
            datasets.val_dataloader(seen=True, unseen=False),
        ]
    )
