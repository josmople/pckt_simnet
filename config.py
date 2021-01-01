import model as M
import utils as U


def config_combination(**kwds):
    from collections import namedtuple
    Config = namedtuple("Config", kwds.keys())
    return U.data.dmap(U.data.dproduct(*kwds.values()), lambda v: Config(*v))


configurations = config_combination(
    models=["simnet", "protonet_1", "protonet_2", "protonet_3"],
    features=["bit", "byte"],
    precision=[16, 32],
    supports=[1, 5, 10, 50],
    classes_seen=[0.4, 0.5, 0.8]
)

models = {
    "protonet_1": lambda: M.ProtonetClassifier(in_channels=416, mid_channels=[], out_channels=32),
    "protonet_2": lambda: M.ProtonetClassifier(in_channels=416, mid_channels=[64], out_channels=32),
    "protonet_3": lambda: M.ProtonetClassifier(in_channels=416, mid_channels=[128, 64], out_channels=32),
    "protonet_4": lambda: M.ProtonetClassifier(in_channels=416, mid_channels=[256, 128, 64], out_channels=32),
}
