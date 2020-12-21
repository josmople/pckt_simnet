import utils as U


class PcapH5Dataset(U.data.ValueDataset):

    def __init__(self, pcap_path, transform=None, h5db_path=None, h5ds_name="all", max_packets=15000):
        self.pcap_path = pcap_path
        self.transform = transform
        self.h5db_path = h5db_path
        self.h5ds_name = h5ds_name
        self.max_packets = max_packets

        self.h5db = None
        super().__init__(self.load_data(), transform)

    def reload(self):
        self.close()
        self.values = self.load_data()

    def load_data(self):
        import os
        import os.path as path

        import scapy.all as sp
        import h5py

        import numpy as np

        from tqdm import tqdm

        if self.h5db_path is None:
            with sp.PcapReader(self.pcap_path) as reader:
                return reader.read_all(self.max_packets)

        if path.exists(self.h5db_path):
            self.h5db = h5py.File(self.h5db_path, "r")
            return self.h5db[self.h5ds_name]

        dirpath = path.dirname(self.h5db_path)
        if len(dirpath) == 0:
            dirpath = "."
        os.makedirs(dirpath, exist_ok=True)

        with h5py.File(self.h5db_path, "w") as db, sp.PcapReader(self.pcap_path) as packets:
            cache = None

            for idx, packet in tqdm(enumerate(packets)):

                if self.max_packets and idx >= self.max_packets:
                    break

                packet.getlayer(0).dst = '0'
                packet.getlayer(0).src = '0'

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

                    if cache is None:
                        dims = num_list.shape
                        cache = db.create_dataset(self.h5ds_name, shape=(1, *dims), maxshape=(None, *dims), chunks=True)

                    if idx >= len(cache):
                        cache.resize(idx + 1, axis=0)
                    cache[idx] = num_list

        self.h5db = h5py.File(self.h5db_path, "r")
        return self.h5db[self.h5ds_name]

    def close(self):
        if self.h5db:
            self.h5db.close()
        self.h5db = None
        self.values = []
