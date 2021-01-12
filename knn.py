from torch.autograd.grad_mode import no_grad
from fewshot import FewshotClassifier
import torch
import typing as T


class KNNClassifier(FewshotClassifier):

    def __init__(self, k: int, no_grad=True):
        super().__init__()
        self.k = k
        self.no_grad = no_grad

    def forward(self, queries: torch.Tensor, *supports: T.List[torch.Tensor]) -> torch.Tensor:
        if self.no_grad:
            with torch.no_grad():
                return self.forward0(queries, *supports)
        return self.forward0(queries, *supports)

    def forward0(self, queries: torch.Tensor, *supports: T.List[torch.Tensor]) -> torch.Tensor:
        distances = []
        for support in supports:
            distances.append(torch.cdist(queries, support))
        distances = torch.cat(distances, dim=1)

        k = min(self.k, distances.size(1))

        _, idxs = distances.topk(k=k)

        labels = []
        for i, s in enumerate(supports):
            labels += [i] * len(s)
        labels = torch.tensor(labels, device=distances.device)

        tops = labels[idxs]

        # _, most_common = tops.mode(dim=1)
        # return torch.eye(len(supports), device=most_common.device)[most_common]

        logits = []
        for row in tops:
            row_logits = torch.bincount(row, minlength=len(supports))
            logits.append(row_logits)
        return torch.stack(logits, dim=0)

        # Batch Bincount
        # zeros = torch.zeros(queries.size(0), len(supports), device=distances.device)
        # ones = torch.ones(queries.size(0), len(supports), device=distances.device)
        # logits = zeros.scatter_add_(dim=1, index=tops, src=ones)
        # return logits
