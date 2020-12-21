import typing as _T

_DB = {
    'AIMchat1': ['aim', 'chat', 'pcapng'],
    'AIMchat2': ['aim', 'chat', 'pcapng'],
    'ICQchat1': ['icq', 'chat', 'pcapng'],
    'ICQchat2': ['icq', 'chat', 'pcapng'],
    'Torrent01': ['torrent', 'pcapng'],
    'aim_chat_3a': ['aim', 'chat', 'pcap'],
    'aim_chat_3b': ['aim', 'chat', 'pcap'],
    'email1a': ['email', 'pcap'],
    'email1b': ['email', 'pcap'],
    'email2a': ['email', 'pcap'],
    'email2b': ['email', 'pcap'],
    'facebook_audio1a': ['facebook', 'audio', 'pcap'],
    'facebook_audio1b': ['facebook', 'audio', 'pcapng'],
    'facebook_audio2a': ['facebook', 'audio', 'pcap'],
    'facebook_audio2b': ['facebook', 'audio', 'pcapng'],
    'facebook_audio3': ['facebook', 'audio', 'pcapng'],
    'facebook_audio4': ['facebook', 'audio', 'pcapng'],
    'facebook_chat_4a': ['facebook', 'chat', 'pcap'],
    'facebook_chat_4b': ['facebook', 'chat', 'pcap'],
    'facebook_video1a': ['facebook', 'video', 'pcap'],
    'facebook_video1b': ['facebook', 'video', 'pcapng'],
    'facebook_video2a': ['facebook', 'video', 'pcap'],
    'facebook_video2b': ['facebook', 'video', 'pcapng'],
    'facebookchat1': ['facebook', 'chat', 'pcapng'],
    'facebookchat2': ['facebook', 'chat', 'pcapng'],
    'facebookchat3': ['facebook', 'chat', 'pcapng'],
    'ftps_down_1a': ['ftps', 'down', 'pcap'],
    'ftps_down_1b': ['ftps', 'down', 'pcap'],
    'ftps_up_2a': ['ftps', 'up', 'pcap'],
    'ftps_up_2b': ['ftps', 'up', 'pcap'],
    'gmailchat1': ['gmail', 'email', 'chat', 'pcapng'],
    'gmailchat2': ['gmail', 'email', 'chat', 'pcapng'],
    'gmailchat3': ['gmail', 'email', 'chat', 'pcapng'],
    'hangout_chat_4b': ['hangouts', 'chat', 'pcap'],
    'hangouts_audio1a': ['hangouts', 'audio', 'pcap'],
    'hangouts_audio1b': ['hangouts', 'audio', 'pcapng'],
    'hangouts_audio2a': ['hangouts', 'audio', 'pcap'],
    'hangouts_audio2b': ['hangouts', 'audio', 'pcapng'],
    'hangouts_audio3': ['hangouts', 'audio', 'pcapng'],
    'hangouts_audio4': ['hangouts', 'audio', 'pcapng'],
    'hangouts_chat_4a': ['hangouts', 'chat', 'pcap'],
    'hangouts_video1b': ['hangouts', 'video', 'pcapng'],
    'hangouts_video2a': ['hangouts', 'video', 'pcap'],
    'hangouts_video2b': ['hangouts', 'video', 'pcapng'],
    'icq_chat_3a': ['icq', 'chat', 'pcap'],
    'icq_chat_3b': ['icq', 'chat', 'pcap'],
    'netflix1': ['netflix', 'pcap'],
    'netflix2': ['netflix', 'pcap'],
    'netflix3': ['netflix', 'pcap'],
    'netflix4': ['netflix', 'pcap'],
    'scp1': ['scp', 'pcapng'],
    'scpDown1': ['scp', 'down', 'pcap'],
    'scpDown2': ['scp', 'down', 'pcap'],
    'scpDown3': ['scp', 'down', 'pcap'],
    'scpDown4': ['scp', 'down', 'pcap'],
    'scpDown5': ['scp', 'down', 'pcap'],
    'scpDown6': ['scp', 'down', 'pcap'],
    'scpUp1': ['scp', 'up', 'pcap'],
    'scpUp2': ['scp', 'up', 'pcap'],
    'scpUp3': ['scp', 'up', 'pcap'],
    'scpUp5': ['scp', 'up', 'pcap'],
    'scpUp6': ['scp', 'up', 'pcap'],
    'sftp1': ['sftp', 'pcapng'],
    'sftpDown1': ['sftp', 'down', 'pcap'],
    'sftpDown2': ['sftp', 'down', 'pcap'],
    'sftpUp1': ['sftp', 'up', 'pcap'],
    'sftp_down_3a': ['sftp', 'down', 'pcap'],
    'sftp_down_3b': ['sftp', 'down', 'pcap'],
    'sftp_up_2a': ['sftp', 'up', 'pcap'],
    'sftp_up_2b': ['sftp', 'up', 'pcap'],
    'skype_audio1a': ['skype', 'audio', 'pcap'],
    'skype_audio1b': ['skype', 'audio', 'pcapng'],
    'skype_audio2a': ['skype', 'audio', 'pcap'],
    'skype_audio2b': ['skype', 'audio', 'pcapng'],
    'skype_audio3': ['skype', 'audio', 'pcapng'],
    'skype_audio4': ['skype', 'audio', 'pcapng'],
    'skype_chat1a': ['skype', 'chat', 'pcap'],
    'skype_chat1b': ['skype', 'chat', 'pcap'],
    'skype_file1': ['skype', 'file', 'pcap'],
    'skype_file2': ['skype', 'file', 'pcap'],
    'skype_file3': ['skype', 'file', 'pcap'],
    'skype_file4': ['skype', 'file', 'pcapng'],
    'skype_file5': ['skype', 'file', 'pcapng'],
    'skype_file6': ['skype', 'file', 'pcapng'],
    'skype_file7': ['skype', 'file', 'pcapng'],
    'skype_file8': ['skype', 'file', 'pcapng'],
    'skype_video1a': ['skype', 'video', 'pcap'],
    'skype_video1b': ['skype', 'video', 'pcapng'],
    'skype_video2a': ['skype', 'video', 'pcap'],
    'skype_video2b': ['skype', 'video', 'pcapng'],
    'spotify1': ['spotify', 'pcap'],
    'spotify2': ['spotify', 'pcap'],
    'spotify3': ['spotify', 'pcap'],
    'spotify4': ['spotify', 'pcap'],
    'torFacebook': ['tor', 'facebook', 'pcap'],
    'torGoogle': ['tor', 'google', 'pcap'],
    'torTwitter': ['tor', 'twitter', 'pcap'],
    'torVimeo1': ['tor', 'vimeo', 'pcap'],
    'torVimeo2': ['tor', 'vimeo', 'pcap'],
    'torVimeo3': ['tor', 'vimeo', 'pcap'],
    'torYoutube1': ['tor', 'youtube', 'pcap'],
    'torYoutube2': ['tor', 'youtube', 'pcap'],
    'torYoutube3': ['tor', 'youtube', 'pcap'],
    'vimeo1': ['vimeo', 'pcap'],
    'vimeo2': ['vimeo', 'pcap'],
    'vimeo3': ['vimeo', 'pcap'],
    'vimeo4': ['vimeo', 'pcap'],
    'voipbuster1b': ['voipbuster', 'pcapng'],
    'voipbuster2b': ['voipbuster', 'pcapng'],
    'voipbuster3b': ['voipbuster', 'pcapng'],
    'voipbuster_4a': ['voipbuster', 'pcap'],
    'voipbuster_4b': ['voipbuster', 'pcap'],
    'vpn_aim_chat1a': ['vpn', 'aim', 'chat', 'pcap'],
    'vpn_aim_chat1b': ['vpn', 'aim', 'chat', 'pcap'],
    'vpn_bittorrent': ['vpn', 'torrent', 'pcap'],
    'vpn_email2a': ['vpn', 'email', 'pcap'],
    'vpn_email2b': ['vpn', 'email', 'pcap'],
    'vpn_facebook_audio2': ['vpn', 'facebook', 'audio', 'pcap'],
    'vpn_facebook_chat1a': ['vpn', 'facebook', 'chat', 'pcap'],
    'vpn_facebook_chat1b': ['vpn', 'facebook', 'chat', 'pcap'],
    'vpn_ftps_A': ['vpn', 'ftps', 'pcap'],
    'vpn_ftps_B': ['vpn', 'ftps', 'pcap'],
    'vpn_hangouts_audio1': ['vpn', 'hangouts', 'audio', 'pcap'],
    'vpn_hangouts_audio2': ['vpn', 'hangouts', 'audio', 'pcap'],
    'vpn_hangouts_chat1a': ['vpn', 'hangouts', 'chat', 'pcap'],
    'vpn_hangouts_chat1b': ['vpn', 'hangouts', 'chat', 'pcap'],
    'vpn_icq_chat1a': ['vpn', 'icq', 'chat', 'pcap'],
    'vpn_icq_chat1b': ['vpn', 'icq', 'chat', 'pcap'],
    'vpn_netflix_A': ['vpn', 'netflix', 'pcap'],
    'vpn_sftp_A': ['vpn', 'sftp', 'pcap'],
    'vpn_sftp_B': ['vpn', 'sftp', 'pcap'],
    'vpn_skype_audio1': ['vpn', 'skype', 'audio', 'pcap'],
    'vpn_skype_audio2': ['vpn', 'skype', 'audio', 'pcap'],
    'vpn_skype_chat1a': ['vpn', 'skype', 'chat', 'pcap'],
    'vpn_skype_chat1b': ['vpn', 'skype', 'chat', 'pcap'],
    'vpn_skype_files1a': ['vpn', 'skype', 'files', 'pcap'],
    'vpn_skype_files1b': ['vpn', 'skype', 'files', 'pcap'],
    'vpn_spotify_A': ['vpn', 'spotify', 'pcap'],
    'vpn_vimeo_A': ['vpn', 'vimeo', 'pcap'],
    'vpn_vimeo_B': ['vpn', 'vimeo', 'pcap'],
    'vpn_voipbuster1a': ['vpn', 'voipbuster', 'pcap'],
    'vpn_voipbuster1b': ['vpn', 'voipbuster', 'pcap'],
    'vpn_youtube_A': ['vpn', 'youtube', 'pcap'],
    'youtube1': ['youtube', 'pcap'],
    'youtube2': ['youtube', 'pcap'],
    'youtube3': ['youtube', 'pcap'],
    'youtube4': ['youtube', 'pcap'],
    'youtube5': ['youtube', 'pcap'],
    'youtube6': ['youtube', 'pcap'],
    'youtubeHTML5_1': ['youtube', 'html', 'pcap']
}


def tags() -> _T.Dict[str, _T.List[str]]:
    return _DB


def names() -> _T.List[str]:
    return list(_DB.keys())


def tagsof(name: str) -> _T.List[str]:
    return list(_DB[name])


def find(pos: _T.Union[None, str, _T.List[str]], neg: _T.Union[None, str, _T.List[str]] = None) -> _T.List[str]:
    if pos is None:
        pos = []
    if neg is None:
        neg = []

    if isinstance(pos, str):
        pos = [pos]
    if isinstance(neg, str):
        neg = [neg]

    assert isinstance(pos, (list, tuple))
    assert isinstance(neg, (list, tuple))

    idxs = []
    for names, taglist in tags().items():
        if all([pt in taglist for pt in pos]) and all([nt not in taglist for nt in neg]):
            idxs.append(names)

    return idxs


del _T
