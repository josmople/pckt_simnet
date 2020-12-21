import utils as U
import dsmd.iscxvpn2016 as md

from dataset import PcapH5Dataset

pcap_dir = "D://Datasets/ISCXVPN2016/"
pckt_dir = "D://Datasets/packets-15k/"

for name in md.names():
    ext = "pcap" if "pcap" in md.tagsof(name) else "pcapng"
    pcap_path = U.paths.join(pcap_dir, name) + "." + ext
    pckt_path = U.paths.join(pckt_dir, name) + "-" + ext + ".h5"

    # print(pcap_path, pckt_path)
    d = PcapH5Dataset(pcap_path, pckt_path)
    print(name, len(d))
