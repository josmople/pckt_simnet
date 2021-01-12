import torch
import torch.nn as nn

import model as M


class OtherWork(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 64, 3),
        )


def paramcount(m: nn.Module):
    total = 0
    for p in m.parameters():
        total += p.numel()
    return total


model = OtherWork()
print(model(torch.ones(1, 1, 28, 28)).size())
print(paramcount(model))
print(paramcount(M.Protonet(416, 32, [128, 64])))
