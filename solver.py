from functools import partial
import model as M
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb
import pytorch_lightning.loggers as pllog

from fewshot import *
from dataloader import *


iscxvpn2016_loader = iscxvpn2016(pcap_dir="D://Datasets/ISCXVPN2016/", h5_dir="D://Datasets/packets-15k/", as_bit=True)
ustctfc2016_loader = ustctfc2016(pcap_dir="D://Datasets/USTC-TFC2016/", h5_dir="D://Datasets/USTC-TFC2016-packets/", as_bit=True)

classifiers = {
    "protonet_1": M.ProtonetClassifier(in_channels=416, mid_channels=[], out_channels=32),
    "protonet_2": M.ProtonetClassifier(in_channels=416, mid_channels=[64], out_channels=32),
    "protonet_3": M.ProtonetClassifier(in_channels=416, mid_channels=[128, 64], out_channels=32),
    "protonet_4": M.ProtonetClassifier(in_channels=416, mid_channels=[256, 128, 64], out_channels=32),
}
# classifier = M.SimnetClassifier(in_channels=416)
# classifier = M.SimnetV2Classifier(in_channels=416)
# classifier = M.ProtonetClassifier(in_channels=416, out_channels=10, mid_channels=[])

for classifier_name, classifier in classifiers.items():
    print("----------------------------------------------------------------------------------")
    print(classifier_name)

    solver = FewshotSolver(classifier)

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
        unseen_classes={k: ustctfc2016_loader(k) for k in ustctfc2016_loader.classes()},
        n_classes=5, n_support=10, n_queries=1000
    )

    tb_logger = pllog.TensorBoardLogger("logs/" + classifier_name)
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=1,
        max_epochs=100,
        log_every_n_steps=1,
        precision=16,
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
