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
from torch.utils.data import DataLoader as DL
import torchvision.transforms.functional as T
import torch
from torch.utils.data.sampler import Sampler
# iscxvpn2016_loader = iscxvpn2016(pcap_dir="D:/CompletePCAPs/", h5_dir="D:/packets-50k/", max_packets_on_cache=50000, as_bit=True, verbose=True)
# ustctfc2016_loader = ustctfc2016(pcap_dir="D:/USTC-TFC2016-master/USTC-TFC2016-master/", h5_dir="D://Datasets/USTC-TFC2016-packets-50k/", max_packets_on_cache=50000, as_bit=True, verbose=True)
import os


def InfiniteSamplerIterator(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

# infinite random sampler, that use InfiniteSamplerIterator


class InfiniteSampler(Sampler):
    def __init__(self, size):
        self.size = size

    def __iter__(self):
        return iter(InfiniteSamplerIterator(self.size))

    def __len__(self):
        return self.size


from knn import KNNClassifier

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

    # "protonet_mini_bit": M.ProtonetClassifier(in_channels=416, mid_channels=[], out_channels=10),

    # "simnet_simple": M.SimnetClassifier(in_channels=416, channels=[10]),
    # "simnet_1": M.SimnetClassifier(in_channels=416, channels=[32]),
    # "simnet_2": M.SimnetClassifier(in_channels=416, channels=[64, 32]),
    # "simnet_3": M.SimnetClassifier(in_channels=416, channels=[128, 64, 32]),

    # "reg_protonet_1": M.ProtonetClassifier(in_channels=52, mid_channels=[], out_channels=32),
    # "reg_protonet_2": M.ProtonetClassifier(in_channels=52, mid_channels=[64], out_channels=32),
    # "reg_protonet_3": M.ProtonetClassifier(in_channels=52, mid_channels=[128, 64], out_channels=32),
    # "reg_protonet_4": M.ProtonetClassifier(in_channels=52, mid_channels=[256, 128, 64], out_channels=32),
    # "reg_protonet_bottleneck_end": M.ProtonetClassifier(in_channels=52, mid_channels=[256, 128, 64], out_channels=10),
    # "reg_protonet_bottleneck_mid": M.ProtonetClassifier(in_channels=52, mid_channels=[128, 32, 128], out_channels=32),

    # "reg_simnet_simple": M.SimnetClassifier(in_channels=416, channels=[10]),
    # "reg_simnet_1": M.SimnetClassifier(in_channels=416, channels=[32]),
    # "reg_simnet_2": M.SimnetClassifier(in_channels=416, channels=[64, 32]),
    # "reg_simnet_3": M.SimnetClassifier(in_channels=416, channels=[128, 64, 32]),
}


classifiers_byte = {
    "knn5": KNNClassifier(k=5),
    "knn10": KNNClassifier(k=10),
    # "relation_1": M.RelationNetClassifier(in_channels=416, feature_channels=[10], simnet_channels=[32, 16, 4]),
    # "reg_relation_1": M.RelationNetClassifier(in_channels=416, feature_channels=[10], simnet_channels=[32, 16, 4]),

    # "protonet_1": M.ProtonetClassifier(in_channels=52, mid_channels=[], out_channels=32),
    # "protonet_2": M.ProtonetClassifier(in_channels=52, mid_channels=[64], out_channels=32),
    # "protonet_3": M.ProtonetClassifier(in_channels=52, mid_channels=[128, 64], out_channels=32),
    # "protonet_4": M.ProtonetClassifier(in_channels=52, mid_channels=[256, 128, 64], out_channels=32),
    # "protonet_bottleneck_end": M.ProtonetClassifier(in_channels=52, mid_channels=[256, 128, 64], out_channels=10),
    # "protonet_bottleneck_mid": M.ProtonetClassifier(in_channels=52, mid_channels=[128, 32, 128], out_channels=32),


    # "protonet_mini_byte_1": M.ProtonetClassifier(in_channels=52, mid_channels=[66], out_channels=10),
    # "protonet_mini_byte_2": M.ProtonetClassifier(in_channels=52, mid_channels=[26], out_channels=10),

    # "simnet_simple": M.SimnetClassifier(in_channels=52, channels=[10]),
    # "simnet_1": M.SimnetClassifier(in_channels=52, channels=[32]),
    # "simnet_2": M.SimnetClassifier(in_channels=52, channels=[64, 32]),
    # "simnet_3": M.SimnetClassifier(in_channels=52, channels=[128, 64, 32]),

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


def accuracy_check(
    classifiers,
    logdir,
    as_bit=True,
    as_half=False,
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

    dataset_collections = {
        "seen": {
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
        "unseen": {
            k: ustctfc2016_loader(k) for k in ustctfc2016_loader.metadata.names()
        },
        "vpntest": {
            "vpn": iscxvpn2016_loader("vpn"),
            "novpn": iscxvpn2016_loader(None, "vpn"),
        },
        "maltest": {
            "benign": ustctfc2016_loader(filter(ustctfc2016_loader.metadata.is_benign, ustctfc2016_loader.metadata.names())),
            "malware": ustctfc2016_loader(filter(ustctfc2016_loader.metadata.is_malware, ustctfc2016_loader.metadata.names())),
        }
    }

    for dataset_name, seen_dic in dataset_collections.items():
        dataset_manager = FewshotDatasetManager(
            seen_classes=seen_dic,
            # unseen_classes={k: ustctfc2016_loader(k) for k in ustctfc2016_loader.metadata.names()},
            n_classes=5, n_support=10, n_queries=batch_size,
            all_classes_val=True
        )

        # datasets_alias = {
        #     seen_datasets_name: "seen",
        #     unseen_datasets_name: "unseen",
        #     "vpn,novpn": "vpntest",
        #     "benign,malware": "malwaretest"
        # }

        for classifier_name, classifier in classifiers.items():

            print("----------------------------------------------------------------------------------")
            print(classifier_name)

            ckpt_file = glob(f"{logdir}/" + classifier_name + "/default/**/checkpoints/*.ckpt")
            if len(ckpt_file) == 0:
                print("CHECKPOINT not found:", ckpt_file)
                if len(list(classifier.parameters())) == 0:
                    print("BUT it has no paraeters so ITS OKAY")
                else:
                    continue
            else:
                ckpt_file = ckpt_file[-1]

            if isinstance(classifier, pl.LightningModule):
                solver = classifier
            else:
                solver = FewshotSolver(classifier)

            if isinstance(ckpt_file, str):
                solver.load_state_dict(torch.load(ckpt_file)["state_dict"])
            solver = solver.cuda()

            test_loader_list = []
            proto_loader_list = []

            print("Seen Classes")
            for key in dataset_manager.datasets_test_seen.keys():
                a = dataset_manager.datasets_test_seen[key]
                dataload = DL(a, batch_size=batch_size)
                test_loader_list.append(dataload)
                print(key)
            print("----------------------------------------------------")

            print("Val Classes")
            for key in dataset_manager.datasets_val_seen.keys():
                b = dataset_manager.datasets_val_seen[key]
                diff_loader = iter(DL(b, batch_size=n_support, sampler=InfiniteSampler(len(b))))
                proto_loader_list.append(diff_loader)

            num_classes = len(proto_loader_list)
            predictions = torch.zeros(num_classes, num_classes)

            rootname = f"{'bit' if as_bit else 'byte'}-{'f16' if as_half else 'f32'}-{classifier_name}"
            with torch.no_grad():
                for d_idx in tqdm(range(len(test_loader_list))):
                    for i, d in enumerate(tqdm(test_loader_list[d_idx])):
                        d = d.cuda()
                        support = []
                        for p_list in proto_loader_list:
                            hotdog = p_list.next()
                            support.append(hotdog.cuda())

                        s = solver(d, *support)
                        s = torch.argmax(s, dim=1)
                        s = torch.bincount(s, minlength=num_classes).cpu()
                        predictions[d_idx] = predictions[d_idx] + s

                dirpath = f"./raw_predictions/{dataset_name}"
                os.makedirs(dirpath, exist_ok=True)
                torch.save(predictions, f"{dirpath}/{rootname}.pt")
                with open(f"{dirpath}/{rootname}.txt", "w+") as f:
                    f.write("\n".join(list(dataset_manager.datasets_test_seen.keys())))


# accuracy_check(
#     classifiers=classifiers,
#     logdir="logs_bigdata_regularized",
#     as_bit=True,
#     as_half=False
# )

accuracy_check(
    classifiers=classifiers_byte,
    logdir="logs_byte",
    as_bit=False,
    as_half=False
)
