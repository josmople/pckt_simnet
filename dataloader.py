from dataset import PcapH5Dataset
import iscxvpn2016

import typing as _T
import torch
import numpy as np
import os.path as P


def load_iscxvpn2016(pos: _T.Union[None, str, _T.List[str]], neg: _T.Union[None, str, _T.List[str]], transform: _T.Union[_T.Callable, None], pcap_dir: str, h5_dir: str):
    names = iscxvpn2016.find(pos, neg)
    assert len(names) > 0, f"Dataset not found pos={pos}, neg={neg}"

    datasets = []
    for name in names:
        ext = "pcap" if "pcap" in iscxvpn2016.tagsof(name) else "pcapng"
        pcap_path = P.join(pcap_dir, name) + "." + ext
        h5_path = P.join(h5_dir, name) + "-" + ext + ".h5"

        dataset = PcapH5Dataset(pcap_path, transform=transform, h5db_path=h5_path)
        datasets.append(dataset)

    return sum(datasets[1:], datasets[0])


def load_iscxvpn2016_bit(pos, neg=None, pcap_dir="D://Datasets/ISCXVPN2016/", h5_dir="D://Datasets/packets-15k/"):
    def packet_to_bit(packet_np: np.ndarray):
        packet = str.join("", map(lambda n: f"{n:08b}", packet_np))
        packet = [int(c) for c in packet]
        return torch.tensor(packet, dtype=torch.float32)
    return load_iscxvpn2016(pos, neg, transform=packet_to_bit, pcap_dir=pcap_dir, h5_dir=h5_dir)


def load_iscxvpn2016_byte(pos, neg=None, pcap_dir="D://Datasets/ISCXVPN2016/", h5_dir="D://Datasets/packets-15k/"):
    def packet_to_byte(packet_np: np.ndarray):
        return torch.tensor(packet_np, dtype=torch.float32)
    return load_iscxvpn2016(pos, neg, transform=packet_to_byte, pcap_dir=pcap_dir, h5_dir=h5_dir)
