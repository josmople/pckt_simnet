from torch.utils.data.dataset import Dataset
from dataset import PcapH5Dataset

import typing as T
import os.path as P

import torch
import numpy as np


class PcapH5Loader:

    def __init__(self, pcap_dir: str, h5_dir: str, max_packets_on_cache=15000, as_bit=True, verbose=True):
        self.pcap_dir = pcap_dir
        self.h5_dir = h5_dir
        self.as_bit = as_bit
        self.max_packets_on_cache = max_packets_on_cache
        self.verbose = verbose

    def preprocessing(self, packet):
        import numpy as np
        import scapy.all as sp

        packet.getlayer(0).dst = '0'
        packet.getlayer(0).src = '0'

        if packet.getlayer(1) is not None:
            if packet.getlayer(1).name == 'IPv6':
                packet.getlayer(1).src = "0000::0000:0000:0000:0000"
                packet.getlayer(1).dst = '0000::00'
            else:
                packet.getlayer(1).src = "0"
                packet.getlayer(1).dst = '0'

            if packet.getlayer(2) is not None:
                packet.getlayer(2).sport = 0
                packet.getlayer(2).dport = 0
                hex_bytes = sp.raw(packet)

                if len(hex_bytes) >= 52:
                    num_list = [int(n) for n in hex_bytes][:40 + 12]
                else:
                    missing = 52 - len(hex_bytes)
                    padding = [0] * missing
                    num_list = [int(n) for n in hex_bytes] + padding
                num_list = np.asarray(num_list).astype(int).reshape(-1)
                return num_list

        return None

    def transform(self, packet: np.ndarray):
        if self.as_bit:
            packet = str.join("", map(lambda n: f"{n:08b}", packet))
            packet = [int(c) for c in packet]
        return torch.tensor(packet, dtype=torch.float32)

    def generate_dataset(self, pcap_list, h5_list):
        datasets = []
        for pcap_path, h5_path in zip(pcap_list, h5_list):
            dataset = PcapH5Dataset(
                pcap_path=P.join(self.pcap_dir, pcap_path),
                h5db_path=P.join(self.h5_dir, h5_path),
                transform_postload=self.transform,
                transform_presave=self.preprocessing,
                max_packets=self.max_packets_on_cache,
                verbose=self.verbose,
            )
            datasets.append(dataset)
        return sum(datasets[1:], datasets[0])

    def __call__(self, *args, **kwds) -> Dataset:
        raise NotImplementedError()


class iscxvpn2016(PcapH5Loader):

    import metadata.iscxvpn2016 as metadata

    def __call__(self, pos: T.Union[None, str, T.List[str]] = None, neg: T.Union[None, str, T.List[str]] = None) -> Dataset:
        names = self.metadata.find(pos, neg)
        assert len(names) > 0, f"Dataset not found pos={pos}, neg={neg}"

        pcap_list, h5_list = [], []
        for name in names:
            ext = "pcap" if "pcap" in self.metadata.tagsof(name) else "pcapng"
            pcap_list.append(f"{name}.{ext}")
            h5_list.append(f"{name}-{ext}.h5")

        return self.generate_dataset(pcap_list, h5_list)


class ustctfc2016(PcapH5Loader):

    import metadata.ustctfc2016 as metadata

    def __call__(self, name: str) -> Dataset:

        paths = self.metadata.findpath(name)
        assert len(paths) > 0, f"Dataset not found: {name}"

        pcap_list, h5_list = [], []
        for path in paths:
            pcap_path = P.join(self.pcap_dir, path)
            zip_path = P.splitext(pcap_path)[0].split("-")[0] + ".7z"
            if not P.exists(pcap_path):
                if P.exists(zip_path):
                    raise Exception(f"PCAP file ({pcap_path}) not found but zip file found ({zip_path})")
                raise Exception(f"PCAP file ({pcap_path}) and no zip file found")

            pcap_list.append(path)
            h5_list.append(f"{P.splitext(path)[0]}.h5")

        return self.generate_dataset(pcap_list, h5_list)
