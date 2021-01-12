from numpy.lib.arraysetops import isin
from torch import tensor
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

import typing as T
import utils as U

from knn import KNNClassifier

torch.backends.cudnn.benchmark = True


def accuracy_test(
    classifiers: T.Dict[str, T.Callable[[int], FewshotClassifier]],
    collections: T.Dict[str, T.Dict[str, U.data.Dataset]],
    as_bit=True,
    as_half=False,
    as_cuda=True,
    n_support=10
):

    cast_tensor = (lambda t: t.cuda()) if as_cuda else (lambda t: t)
    cast_integer = (lambda t: torch.tensor(t, device="cuda") if as_cuda else (lambda t: t))

    for classifier_name, classifier_constructor in classifiers.items():
        classifier = classifier_constructor(416 if as_bit else 52)

        if as_half:
            classifier.half()
        if as_cuda:
            classifier.cuda()

        for collection_name, datasets in collections.items():

            confusion_matrix = ConfusionMatrix(num_classes=len(datasets))

            key_list = list(datasets.keys())
            value_list = [datasets[k] for k in key_list]

            queries_ds = U.data.dmap(
                sum(value_list[1:], value_list[0]),
                transform=cast_tensor
            )
            labels_ds = U.data.dmap(
                [x for i, ds in enumerate(value_list) for x in [i] * len(ds)],
                transform=cast_integer
            )

            support_ds_list = []
            for dataset in datasets:
                support_ds = U.data.dconst()


accuracy_test(
    classifiers={

    },
    collections={
        "seen": {},
        "unseen": {},
        "vpntest": {},
        "malware": {},
    }
)


def accuracy_check(
    classifiers,
    logdir,
    as_bit=True,
    as_half=False,
    as_cuda=False,
    iscxvpn2016_pcap="D://Datasets/ISCXVPN2016/",
    iscxvpn2016_h5="D://Datasets/packets-50k/",
    ustctfc2016_pcap="D://Datasets/USTC-TFC2016/",
    ustctfc2016_h5="D://Datasets/USTC-TFC2016-packets-50k/",
    max_packets_on_cache=50000,
    random_seed=2020,
    batch_size=20000,
    n_support=10
):

    iscxvpn2016_loader = iscxvpn2016(pcap_dir=iscxvpn2016_pcap, h5_dir=iscxvpn2016_h5, max_packets_on_cache=max_packets_on_cache, as_bit=as_bit, verbose=True)
    ustctfc2016_loader = ustctfc2016(pcap_dir=ustctfc2016_pcap, h5_dir=ustctfc2016_h5, max_packets_on_cache=max_packets_on_cache, as_bit=as_bit, verbose=True)

    pl.seed_everything(seed=random_seed)
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
        n_classes=5, n_support=n_support, n_queries=batch_size,
        all_classes_val=True
    )
    seen_datasets = dataset_manager.val_dataloader(seen=True, unseen=False)
    seen_datasets_name = ",".join([str(k) for k in dataset_manager.last_datasets.keys()])
    unseen_datasets = dataset_manager.val_dataloader(seen=False, unseen=True)
    unseen_datasets_name = ",".join([str(k) for k in dataset_manager.last_datasets.keys()])

    datasets = {
        seen_datasets_name: seen_datasets,
        unseen_datasets_name: unseen_datasets,
        "vpn,novpn": fewshot_dataloader([
            iscxvpn2016_loader("vpn"),
            iscxvpn2016_loader(None, "vpn"),
        ], n_support=10, n_queries=batch_size),
        "benign,malware": fewshot_dataloader([
            ustctfc2016_loader(filter(ustctfc2016_loader.metadata.is_benign, ustctfc2016_loader.metadata.names())),
            ustctfc2016_loader(filter(ustctfc2016_loader.metadata.is_malware, ustctfc2016_loader.metadata.names())),
        ], n_support=n_support, n_queries=batch_size)
    }
    datasets_alias = {
        seen_datasets_name: "seen",
        unseen_datasets_name: "unseen",
        "vpn,novpn": "vpntest",
        "benign,malware": "malwaretest"
    }

    for classifier_name, classifier in classifiers.items():

        print("----------------------------------------------------------------------------------")
        print(classifier_name)

        ckpt_file = glob(f"{logdir}/" + classifier_name + "/default/**/checkpoints/*.ckpt")
        if len(ckpt_file) <= 0:
            print("CHECKPOINT not found:", ckpt_file)
            continue
        ckpt_file = ckpt_file[-1]

        if isinstance(classifier, pl.LightningModule):
            solver = classifier
        else:
            solver = FewshotSolver(classifier)

        solver.load_state_dict(torch.load(ckpt_file)["state_dict"])
        if as_half:
            solver.half()

        if as_cuda:
            solver.cuda()

        with torch.no_grad():

            for i, (dataset_name, dataset) in enumerate(datasets.items()):
                evaluators = {}
                print("DATASET: ", dataset_name)
                for j, (queries, labels, *supports) in enumerate(tqdm(dataset)):
                    # if j > 20:
                    #     break

                    n_classes = len(supports)
                    if len(evaluators) == 0:
                        evaluators = {
                            "accuracy": plmc.Accuracy(),
                            "precision": plmc.Precision(num_classes=n_classes),
                            "recall": plmc.Recall(num_classes=n_classes),
                            "fbeta": plmc.FBeta(num_classes=n_classes),
                            "f1": plmc.F1(num_classes=n_classes),
                            "confmat": plmc.ConfusionMatrix(num_classes=n_classes)
                        }

                        if as_cuda:
                            for evaluator in evaluators.values():
                                evaluator.cuda()

                    if as_cuda:
                        queries = queries.cuda()
                        supports = [s.cuda() for s in supports]

                    if as_half:
                        logits = solver(queries.half(), *[s.half() for s in supports])
                    else:
                        logits = solver(queries, *supports)

                    for evaluator_name, evaluator in evaluators.items():
                        evaluator(logits.float(), labels.cuda() if as_cuda else labels)

                rootname = f"{'bit' if as_bit else 'byte'}-{'f16' if as_half else 'f32'}-{classifier_name}-{datasets_alias[dataset_name]}"
                outputs = {}
                for evaluator_name, evaluator in evaluators.items():

                    if evaluator_name != "confmat":
                        val = evaluator.compute().cpu()
                        outputs[evaluator_name] = val.item()
                        print(evaluator_name, val)
                    else:
                        matrix = evaluator.compute().cpu()
                        plotconfmat(dataset_name.split(","), matrix, rootname + ".png")
                        torch.save(matrix, rootname + ".pt")
                        # T.to_pil_image(matrix).save(f"matrix_{classifier_name}_dataset{i:02}.png")

                import json
                with open(rootname + ".json", "w+") as f:
                    json.dump(outputs, f)

        solver.cpu()


classifiers = {
    "knn5": KNNClassifier(k=5),
    "knn10": KNNClassifier(k=10),

    # "relpnet_1": M.RelationNetClassifier_Protonet1(simnet_channels=[128, 64, 32]),

    # "relation_1": M.RelationNetClassifier(in_channels=416, feature_channels=[10], simnet_channels=[32, 16, 4]),
    # "reg_relation_1": M.RelationNetClassifier(in_channels=416, feature_channels=[10], simnet_channels=[32, 16, 4]),

    # "protonet_1": M.ProtonetClassifier(in_channels=416, mid_channels=[], out_channels=32),
    # "protonet_2": M.ProtonetClassifier(in_channels=416, mid_channels=[64], out_channels=32),
    # "protonet_3": M.ProtonetClassifier(in_channels=416, mid_channels=[128, 64], out_channels=32),
    # "protonet_4": M.ProtonetClassifier(in_channels=416, mid_channels=[256, 128, 64], out_channels=32),
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
accuracy_check(
    classifiers=classifiers,
    logdir="logs_bigdata_regularized",
    as_bit=True,
    as_half=False
)

print("DOING bytes!!!!")

classifiers = {
    "knn5": KNNClassifier(k=5),
    "knn10": KNNClassifier(k=10),

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
accuracy_check(
    classifiers=classifiers,
    logdir="logs_byte",
    as_bit=False,
    as_half=False
)
