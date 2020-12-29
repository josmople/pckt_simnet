import utils as U


class PcapH5Dataset(U.data.ValueDataset):

    def __init__(self, pcap_path, transform=None, h5db_path=None, h5ds_name="all", max_packets=15000, verbose=True, preprocessing=None):
        self.pcap_path = pcap_path
        self.transform = transform
        self.h5db_path = h5db_path
        self.h5ds_name = h5ds_name
        self.max_packets = max_packets
        self.verbose = verbose

        if preprocessing:
            self.preprocessing = preprocessing

        self.h5db = None
        super().__init__(self.load_data(), transform)

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

                packet_numpy = self.preprocessing(packet)

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
