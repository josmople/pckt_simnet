import torch
import torch.nn as nn

import typing as T


class FewshotClassifier(nn.Module):

    def __call__(self, queries: torch.Tensor, *supports: T.List[torch.Tensor]) -> torch.Tensor:
        return super().__call__(queries, *supports)

    def forward(self, queries: torch.Tensor, *supports: T.List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()
