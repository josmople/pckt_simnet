import utils as U


class PcapH5Dataset(U.data.ValueDataset):

    def __init__(self, pcap_path, h5db_path=None, transform_presave=None, transform_postload=None, h5ds_name="all", max_packets=15000, verbose=True):
        self.pcap_path = pcap_path

        self.h5db = None
        self.h5db_path = h5db_path
        self.h5ds_name = h5ds_name

        self.max_packets = max_packets
        self.verbose = verbose

        self.preprocess = transform_presave or self.default_transform_presave
        super().__init__(self.load_data(), transform_postload)

    def default_transform_presave(self, packet):
        from scapy.packet import raw
        from numpy import asarray
        hex_bytes = raw(packet)
        int_bytes = [int(n) for n in hex_bytes]
        return asarray(int_bytes).astype(int).reshape(-1)

    def reload(self):
        self.close()
        self.values = self.load_data()

    def load_data(self):
        import os
        import os.path as path

        import numpy as np
        import scapy.all as sp
        import h5py as hp

        from tqdm import tqdm

        if self.h5db_path is None:
            with sp.PcapReader(self.pcap_path) as reader:
                return reader.read_all(self.max_packets)

        if path.exists(self.h5db_path):
            if self.verbose:
                print(f"Loading: pcap='{self.pcap_path}' h5='{self.h5db_path}'")
            self.h5db = hp.File(self.h5db_path, "r")
            return self.h5db[self.h5ds_name]

        dirpath = path.dirname(self.h5db_path)
        if len(dirpath) == 0:
            dirpath = "."
        os.makedirs(dirpath, exist_ok=True)

        with hp.File(self.h5db_path, "w") as db, sp.PcapReader(self.pcap_path) as packets:

            if self.verbose:
                print(f"Caching: pcap='{self.pcap_path}' h5='{self.h5db_path}'")
            cache = None

            processed_packets = 0

            iterable = enumerate(packets)
            if self.verbose:
                iterable = tqdm(iterable)

            for idx, packet in iterable:

                if self.max_packets and processed_packets >= self.max_packets:
                    break

                packet_numpy = self.preprocess(packet)

                if packet_numpy is None:
                    continue
                processed_packets += 1

                if cache is None:
                    dims = packet_numpy.shape
                    cache = db.create_dataset(self.h5ds_name, shape=(1, *dims), maxshape=(None, *dims), dtype=np.int, chunks=True)

                if processed_packets >= len(cache):
                    cache.resize(processed_packets + 1, axis=0)
                    cache[processed_packets] = packet_numpy

        self.h5db = hp.File(self.h5db_path, "r")
        return self.h5db[self.h5ds_name]

    def close(self):
        if self.h5db:
            self.h5db.close()
        self.h5db = None
        self.values = []
