def ts(template="%Y-%m-%d %H-%M-%S.%f"):
    from datetime import datetime
    now = datetime.now()
    path = now.strftime(template)
    return path


def join(*paths):
    from os.path import join as _join
    return _join(*[str(p) for p in paths])


def makedir(path):
    from os import makedirs
    from os.path import dirname
    pathdir = dirname(path)
    if pathdir == "":
        pathdir = "."
    makedirs(pathdir, exist_ok=True)
    return pathdir


def dirpath(*paths, auto_makedir=True):
    basepath = join(*paths)

    if auto_makedir:
        def resolve_path(*paths):
            path = join(basepath, *paths)
            makedir(path)
            return path
    else:
        def resolve_path(*paths):
            path = join(basepath, *paths)
            return path

    return resolve_path
