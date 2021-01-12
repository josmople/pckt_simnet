import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from glob import glob


import model as M


def paramcount(m: nn.Module):
    total = 0
    for p in m.parameters():
        total += p.numel()
    return total


class Net(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 64, 3),
            # Some ReLU and Max-Pool
            nn.Conv2d(64, 64, 3),
            # Some ReLU and Max-Pool
            nn.Conv2d(64, 64, 3),
            # Some ReLU and Max-Pool
        )


a = M.Protonet(416, 32, mid_channels=[])
# b = M.Protonet(52, 10, mid_channels=[66])
c = Net()

print(paramcount(a))
# print(paramcount(b))
print(paramcount(c))
