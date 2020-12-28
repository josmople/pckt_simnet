from torch.utils.data.dataset import Dataset
from dataset import PcapH5Dataset

import typing as T
import os.path as P

import torch
import numpy as np


class iscxvpn2016:
    import metadata.iscxvpn2016 as metadata

    def __init__(self, pcap_dir: str, h5_dir: str, as_bit=True, max_packets_on_cache=15000):
        self.pcap_dir = pcap_dir
        self.h5_dir = h5_dir
        self.as_bit = as_bit
        self.max_packets_on_cache = max_packets_on_cache

    def transform(self, packet: np.ndarray):
        if self.as_bit:
            packet = str.join("", map(lambda n: f"{n:08b}", packet))
            packet = [int(c) for c in packet]
        return torch.tensor(packet, dtype=torch.float32)

    def classes(self):
        return self.metadata.names()

    def __call__(self, pos: T.Union[None, str, T.List[str]] = None, neg: T.Union[None, str, T.List[str]] = None) -> Dataset:
        names = self.metadata.find(pos, neg)
        assert len(names) > 0, f"Dataset not found pos={pos}, neg={neg}"

        datasets = []
        for name in names:
            ext = "pcap" if "pcap" in self.metadata.tagsof(name) else "pcapng"
            pcap_path = P.join(self.pcap_dir, name) + "." + ext
            h5_path = P.join(self.h5_dir, name) + "-" + ext + ".h5"

            dataset = PcapH5Dataset(pcap_path, transform=self.transform, h5db_path=h5_path, max_packets=self.max_packets_on_cache)
            datasets.append(dataset)

        return sum(datasets[1:], datasets[0])


class ustctfc2016:
    import metadata.ustctfc2016 as metadata

    def __init__(self, pcap_dir: str, h5_dir: str, as_bit=True, max_packets_on_cache=15000):
        self.pcap_dir = pcap_dir
        self.h5_dir = h5_dir
        self.as_bit = as_bit
        self.max_packets_on_cache = max_packets_on_cache

    def transform(self, packet: np.ndarray):
        if self.as_bit:
            packet = str.join("", map(lambda n: f"{n:08b}", packet))
            packet = [int(c) for c in packet]
        return torch.tensor(packet, dtype=torch.float32)

    def classes(self):
        return self.metadata.names()

    def __call__(self, name: str) -> Dataset:
        paths = self.metadata.findpath(name)
        assert len(paths) > 0, f"Dataset not found: {name}"

        datasets = []
        for name in paths:
            pcap_path = P.join(self.pcap_dir, name)
            zip_path = P.splitext(pcap_path)[0].split("-")[0] + ".7z"
            h5_path = P.splitext(P.join(self.h5_dir, name))[0] + ".h5"

            if not P.exists(pcap_path):
                if P.exists(zip_path):
                    raise Exception(f"PCAP file ({pcap_path}) not found but zip file found ({zip_path})")
                raise Exception(f"PCAP file ({pcap_path}) and no zip file found")

            dataset = PcapH5Dataset(pcap_path, transform=self.transform, h5db_path=h5_path, max_packets=self.max_packets_on_cache)
            datasets.append(dataset)

        return sum(datasets[1:], datasets[0])
