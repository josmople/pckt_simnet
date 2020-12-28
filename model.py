import torch
import torch.nn as nn

import typing as _T

from fewshot import FewshotClassifier


def fully_connected(channels, activation=None, bias=True):

    in_channels = channels[:-1]  # Except Last
    out_channels = channels[1:]  # Except First

    if not isinstance(activation, (list, tuple)):
        activation = [activation] * len(in_channels)
    assert len(activation) == len(in_channels)
    assert all([a is None or callable(a) for a in activation])

    if not isinstance(bias, (list, tuple)):
        bias = [bias] * len(in_channels)
    assert len(bias) == len(in_channels)
    bias = [bool(b) for b in bias]

    layers = []
    for i, o, b, a in zip(in_channels, out_channels, bias, activation):
        layers.append(nn.Linear(in_features=i, out_features=o, bias=b))
        if a is not None:
            layers.append(a())

    return layers


class Simnet(nn.Sequential):
    def __init__(self, in_channels, channels=[256, 128, 64, 10]):
        layers = fully_connected([in_channels * 2] + channels, activation=nn.ReLU, bias=True)[:-1]
        super().__init__(
            *layers,
            nn.Tanh()
        )

    def forward(self, a, b):
        x = torch.cat([a, b], dim=1)
        return super().forward(x).mean(dim=1)


class SimnetClassifier(FewshotClassifier):

    def __init__(self, in_channels, channels=[256, 128, 64, 10]):
        super().__init__()
        self.simnet = Simnet(in_channels=in_channels, channels=channels)

    def __call__(self, queries: torch.Tensor, *supports: _T.List[torch.Tensor]):
        return super().__call__(queries, *supports)

    def forward(self, queries: torch.Tensor, *supports: _T.List[torch.Tensor]):
        assert queries.dim() == 2

        num_query, num_dim = queries.size()
        num_classes = len(supports)

        assert all([class_supports.dim() == 2 for class_supports in supports])
        assert all([class_supports.size(1) == num_dim for class_supports in supports])

        scores = []
        for class_supports in supports:
            class_score = 0
            num_support = class_supports.size(0)
            for i in range(num_support):
                item = class_supports[i]
                item = item.unsqueeze(0).repeat(num_query, 1)
                item_score = self.simnet(queries, item)
                if item_score.dim() == 1:
                    item_score.unsqueeze_(1)
                assert item_score.dim() == 2
                class_score += item_score
            scores.append(class_score)
            assert class_score.dim() == 2

        scores = torch.cat(scores, dim=1)
        assert scores.size() == (num_query, num_classes)
        return scores


class SimnetV2Classifier(FewshotClassifier):

    def __init__(self, in_channels, feature_channels=[128, 64, 32], simnet_channels=[128, 64, 32]):
        super().__init__()
        channels = [in_channels, *feature_channels]
        self.features = nn.Sequential(*fully_connected(channels, activation=nn.ReLU, bias=True)[:-1])
        self.simnet = Simnet(in_channels=channels[-1], channels=simnet_channels)

    def __call__(self, queries: torch.Tensor, *supports: _T.List[torch.Tensor]):
        return super().__call__(queries, *supports)

    def forward(self, queries: torch.Tensor, *supports: _T.List[torch.Tensor]):
        assert queries.dim() == 2

        num_query, num_dim = queries.size()
        num_classes = len(supports)

        assert all([class_supports.dim() == 2 for class_supports in supports])
        assert all([class_supports.size(1) == num_dim for class_supports in supports])

        scores = []
        for class_supports in supports:
            class_score = 0
            num_support = class_supports.size(0)
            for i in range(num_support):
                item = class_supports[i]
                item = item.unsqueeze(0)

                item_features = self.features(item)
                item_features = item_features.repeat(num_query, 1)

                queries_features = self.features(queries)

                item_score = self.simnet(queries_features, item_features)
                if item_score.dim() == 1:
                    item_score.unsqueeze_(1)
                assert item_score.dim() == 2
                class_score += item_score
            scores.append(class_score)
            assert class_score.dim() == 2

        scores = torch.cat(scores, dim=1)
        assert scores.size() == (num_query, num_classes)
        return scores


class Protonet(nn.Sequential):

    def __init__(self, in_channels, out_channels, mid_channels=[256, 128, 64]):
        layers = fully_connected([in_channels, *mid_channels, out_channels], activation=nn.ReLU, bias=True)[:-1]
        super().__init__(*layers)


class ProtonetClassifier(FewshotClassifier):

    def __init__(self, in_channels, out_channels, mid_channels=[256, 128, 64]):
        super().__init__()
        self.protonet = Protonet(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)

    def compute_prototype(self, features: torch.Tensor):
        return features.mean(dim=0, keepdim=True)

    def pairwise_distance(self, x, y):
        assert x.dim() == 2
        assert y.dim() == 2
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        return torch.cdist(x, y, p=2).squeeze(0)

    def forward(self, queries, *supports):
        assert queries.dim() == 2

        num_query, num_dim = queries.size()
        num_classes = len(supports)

        assert all([class_supports.dim() == 2 for class_supports in supports])
        assert all([class_supports.size(1) == num_dim for class_supports in supports])

        queries = self.protonet(queries)
        num_query, num_feat = queries.size()

        prototypes = []
        for class_support in supports:
            prototype = self.protonet(class_support)
            assert prototype.size(1) == num_feat

            prototype = self.compute_prototype(prototype)
            assert prototype.size() == (1, num_feat)

            prototypes.append(prototype)

        prototypes = torch.cat(prototypes, dim=0)
        assert prototypes.size() == (num_classes, num_feat)

        scores = -self.pairwise_distance(queries, prototypes)
        assert scores.size() == (num_query, num_classes)
        return scores
