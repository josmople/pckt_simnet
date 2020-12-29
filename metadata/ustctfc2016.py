import typing as _T

_DB = {
    "BitTorrent": ["Benign/BitTorrent.pcap"],
    "Facetime": ["Benign/Facetime.pcap"],
    "FTP": ["Benign/FTP.pcap"],
    "Gmail": ["Benign/Gmail.pcap"],
    "MySQL": ["Benign/MySQL.pcap"],
    "Outlook": ["Benign/Outlook.pcap"],
    "Skype": ["Benign/Skype.pcap"],
    "SMB": ["Benign/SMB-1.pcap", "Benign/SMB-2.pcap"],
    "Weibo": ["Benign/Weibo-1.pcap", "Benign/Weibo-2.pcap", "Benign/Weibo-3.pcap", "Benign/Weibo-4.pcap"],
    "WorldOfWarcraft": ["Benign/WorldOfWarcraft.pcap"],
    "Cridex": ["Malware/Cridex.pcap"],
    "Geodo": ["Malware/Geodo.pcap"],
    "Htbot": ["Malware/Htbot.pcap"],
    "Miuref": ["Malware/Miuref.pcap"],
    "Neris": ["Malware/Neris.pcap"],
    "Nsis-ay": ["Malware/Nsis-ay.pcap"],
    "Shifu": ["Malware/Shifu.pcap"],
    "Tinba": ["Malware/Tinba.pcap"],
    "Virut": ["Malware/Virut.pcap"],
    "Zeus": ["Malware/Zeus.pcap"]
}


def load_aliases():
    aliases = {}

    for key, val in list(_DB.items()):
        aliases[key] = key
        for path in val:
            aliases[path] = key

    aliases["Torrent"] = "BitTorrent"
    aliases["SQL"] = "MySQL"
    aliases["WOW"] = "WorldOfWarcraft"
    aliases["Nsis"] = "Nsis-ay"

    for key, val in list(aliases.items()):
        aliases[key.lower()] = val

    return aliases


_ALIASES = load_aliases()


def names(include_benign=True, include_malware=True) -> _T.List[str]:
    if include_benign and include_malware:
        return list(_DB.keys())
    if include_benign:
        return list(filter(is_benign, _DB.keys()))
    if include_malware:
        return list(filter(is_malware, _DB.keys()))
    return []


def findpath(name: str) -> _T.List[str]:
    try:
        return _DB[_ALIASES[name.lower()]]
    except IndexError:
        return []


def is_malware(name: str):
    paths = _DB[_ALIASES[name.lower()]]
    assert len(paths) > 0
    return "Malware" in paths[0]


def is_benign(name: str):
    paths = _DB[_ALIASES[name.lower()]]
    assert len(paths) > 0
    return "Benign" in paths[0]


del _T
