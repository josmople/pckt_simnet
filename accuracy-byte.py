from numpy.lib.arraysetops import isin
import model as M
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb
import pytorch_lightning.trainer as pltr
import pytorch_lightning.loggers as pllog

from fewshot import *
from dataloader import *
from accuracy_utils import *

from pytorch_lightning.metrics import Accuracy, Precision, Recall, ConfusionMatrix, F1

from glob import glob
from tqdm import tqdm

import torchvision.transforms.functional as T


iscxvpn2016_loader = iscxvpn2016(pcap_dir="D://Datasets/ISCXVPN2016/", h5_dir="D://Datasets/packets-50k/", max_packets_on_cache=50000, as_bit=False, verbose=True)
ustctfc2016_loader = ustctfc2016(pcap_dir="D://Datasets/USTC-TFC2016/", h5_dir="D://Datasets/USTC-TFC2016-packets-50k/", max_packets_on_cache=50000, as_bit=False, verbose=True)


classifiers = {

    # "relpnet_1": M.RelationNetClassifier_Protonet1(simnet_channels=[128, 64, 32]),
    # "protonet_mini_byte_1": M.ProtonetClassifier(in_channels=52, mid_channels=[66], out_channels=10),
    # "protonet_mini_byte_2": M.ProtonetClassifier(in_channels=52, mid_channels=[26], out_channels=10),

    # "relation_1": M.RelationNetClassifier(in_channels=416, feature_channels=[10], simnet_channels=[32, 16, 4]),
    # "reg_relation_1": M.RelationNetClassifier(in_channels=416, feature_channels=[10], simnet_channels=[32, 16, 4]),

    # "protonet_1": M.ProtonetClassifier(in_channels=52, mid_channels=[], out_channels=32),
    # "protonet_2": M.ProtonetClassifier(in_channels=52, mid_channels=[64], out_channels=32),
    # "protonet_3": M.ProtonetClassifier(in_channels=52, mid_channels=[128, 64], out_channels=32),
    # "protonet_4": M.ProtonetClassifier(in_channels=52, mid_channels=[256, 128, 64], out_channels=32),
    # "protonet_bottleneck_end": M.ProtonetClassifier(in_channels=416, mid_channels=[256, 128, 64], out_channels=10),
    # "protonet_bottleneck_mid": M.ProtonetClassifier(in_channels=416, mid_channels=[128, 32, 128], out_channels=32),

    # "simnet_simple": M.SimnetClassifier(in_channels=416, channels=[10]),
    # "simnet_1": M.SimnetClassifier(in_channels=416, channels=[32]),
    # "simnet_2": M.SimnetClassifier(in_channels=416, channels=[64, 32]),
    # "simnet_3": M.SimnetClassifier(in_channels=416, channels=[128, 64, 32]),

    # "reg_protonet_1": M.ProtonetClassifier(in_channels=416, mid_channels=[], out_channels=32),
    # "reg_protonet_2": M.ProtonetClassifier(in_channels=416, mid_channels=[64], out_channels=32),
    # "reg_protonet_3": M.ProtonetClassifier(in_channels=416, mid_channels=[128, 64], out_channels=32),
    # "reg_protonet_4": M.ProtonetClassifier(in_channels=416, mid_channels=[256, 128, 64], out_channels=32),
    # "reg_protonet_bottleneck_end": M.ProtonetClassifier(in_channels=416, mid_channels=[256, 128, 64], out_channels=10),
    # "reg_protonet_bottleneck_mid": M.ProtonetClassifier(in_channels=416, mid_channels=[128, 32, 128], out_channels=32),

    # "reg_simnet_simple": M.SimnetClassifier(in_channels=416, channels=[10]),
    # "reg_simnet_1": M.SimnetClassifier(in_channels=416, channels=[32]),
    # "reg_simnet_2": M.SimnetClassifier(in_channels=416, channels=[64, 32]),
    # "reg_simnet_3": M.SimnetClassifier(in_channels=416, channels=[128, 64, 32]),
}


pl.seed_everything(seed=2020)
dataset_manager = FewshotDatasetManager(
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
    n_classes=5, n_support=10, n_queries=1000,
    all_classes_val=True
)
seen_datasets = dataset_manager.val_dataloader(seen=True, unseen=False)
seen_datasets_name = ",".join([str(k) for k in dataset_manager.last_datasets.keys()])
unseen_datasets = dataset_manager.val_dataloader(seen=False, unseen=True)
unseen_datasets_name = ",".join([str(k) for k in dataset_manager.last_datasets.keys()])

datasets = {
    seen_datasets_name: seen_datasets,
    unseen_datasets_name: unseen_datasets
}


for classifier_name, classifier in classifiers.items():

    print("----------------------------------------------------------------------------------")
    print(classifier_name)

    ckpt_file = glob("logs_byte/" + classifier_name + "/default/**/checkpoints/*.ckpt")
    if len(ckpt_file) < 0:
        print("CHECKPOINT not found:", ckpt_file)
        continue
    ckpt_file = ckpt_file[-1]

    if isinstance(classifier, pl.LightningModule):
        solver = classifier
    else:
        solver = FewshotSolver(classifier)

    trainer = pl.Trainer(solver)
    solver.load_state_dict(torch.load(ckpt_file)["state_dict"])
    solver.cpu()
    # solver.load_from_checkpoint(ckpt_file)

    # print("Ok")

    for i, (dataset_name, dataset) in enumerate(datasets.items()):

        n_classes = len(dataset_manager.datasets_val_seen)

        evaluators = {}

        print("DATASET: ", dataset_name)
        for queries, labels, *supports in tqdm(dataset):

            n_classes = len(supports)
            if len(evaluators) == 0:
                evaluators = {
                    "accuracy": plmc.Accuracy(),
                    "precision": plmc.Precision(num_classes=n_classes),
                    "recall": plmc.Recall(num_classes=n_classes),
                    "fbeta": plmc.FBeta(num_classes=n_classes),
                    "f1": plmc.F1(num_classes=n_classes),
                    "confmat": plmc.ConfusionMatrix(num_classes=n_classes, normalize="true")
                }
                for evaluator in evaluators.values():
                    evaluator.cpu()

            logits = solver(queries.cpu(), *[s.cpu() for s in supports])
            for evaluator_name, evaluator in evaluators.items():
                evaluator(logits, labels.cpu())

        for evaluator_name, evaluator in evaluators.items():

            if evaluator_name != "confmat":
                print("METRIC: ", evaluator_name)
                print(evaluator.compute())
            else:
                matrix = evaluator.compute()
                plotconfmat(dataset_name.split(","), matrix, f"matrix-byte-{classifier_name}-dataset{i:02}.png")
                # T.to_pil_image(matrix).save(f"matrix_{classifier_name}_dataset{i:02}.png")

    solver.cpu()

    print("Oks")
