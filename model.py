import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import typing as _T

from fewshot import FewshotClassifier

import pytorch_lightning as pl
import typing as T
import pytorch_lightning.metrics as plmc


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


class RelationNetClassifier(pl.LightningModule, FewshotClassifier):

    def __init__(self, in_channels, feature_channels=[128, 64, 32], simnet_channels=[128, 64, 32], lr=1e-3, weight_decay=1e-4, lambda_metric=0.1):
        pl.LightningModule.__init__(self)

        channels = [in_channels, *feature_channels]
        self.features = nn.Sequential(*fully_connected(channels, activation=nn.ReLU, bias=True)[:-1])
        self.simnet = Simnet(in_channels=channels[-1], channels=simnet_channels)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_metric = lambda_metric
        self.evaluators = nn.ModuleDict()

    def forward(self, queries: torch.Tensor, *supports: _T.List[torch.Tensor]):
        assert queries.dim() == 2

        num_query, num_dim = queries.size()
        num_classes = len(supports)

        assert all([class_supports.dim() == 2 for class_supports in supports])
        assert all([class_supports.size(1) == num_dim for class_supports in supports])

        queries_features = self.features(queries)

        scores = []
        for class_supports in supports:
            class_score = 0
            num_support = class_supports.size(0)
            for i in range(num_support):
                item = class_supports[i]
                item = item.unsqueeze(0)

                item_features = self.features(item)
                item_features = item_features.repeat(num_query, 1)

                item_score = self.simnet(queries_features, item_features)
                if item_score.dim() == 1:
                    item_score.unsqueeze_(1)

                assert item_score.dim() == 2
                class_score += item_score

            assert class_score.dim() == 2
            scores.append(class_score)

        scores = torch.cat(scores, dim=1)
        assert scores.size() == (num_query, num_classes)
        return scores

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def validation_step(self, batch: T.List[torch.Tensor], batch_idx: int, dataloader_idx: int):
        queries, labels, *supports = batch
        logits = self(queries, *supports)

        if dataloader_idx not in self.evaluators:
            eval_n_classes = len(supports)
            self.evaluators[f"dl_{dataloader_idx}"] = nn.ModuleDict({
                "accuracy": plmc.Accuracy(),
                "precision": plmc.Precision(num_classes=eval_n_classes),
                "recall": plmc.Recall(num_classes=eval_n_classes),
                "fbeta": plmc.FBeta(num_classes=eval_n_classes),
                "f1": plmc.F1(num_classes=eval_n_classes),
                "confmat": plmc.ConfusionMatrix(num_classes=eval_n_classes)
            }).to(device=self.device)

        evaluators = self.evaluators[f"dl_{dataloader_idx}"]
        for category, evaluator in evaluators.items():
            self.log(f"metrics/{category}", evaluator(logits, labels))

    def training_step(self, batch: T.List[torch.Tensor], batch_idx: int):
        queries, labels, *supports = batch

        assert queries.dim() == 2

        num_query, num_dim = queries.size()
        num_classes = len(supports)

        assert all([class_supports.dim() == 2 for class_supports in supports])
        assert all([class_supports.size(1) == num_dim for class_supports in supports])

        queries_features = self.features(queries)

        logits = []
        metric_loss = 0
        for class_idx, class_supports in enumerate(supports):
            class_score = 0
            # metric_score = 0
            num_support = class_supports.size(0)
            for i in range(num_support):
                item = class_supports[i]
                item = item.unsqueeze(0)

                item_features = self.features(item)
                item_features = item_features.repeat(num_query, 1)

                item_score = self.simnet(queries_features, item_features)
                if item_score.dim() == 1:
                    item_score.unsqueeze_(1)

                assert item_score.dim() == 2
                class_score += item_score

                # metric_score += item_score[labels == class_idx]

            assert class_score.dim() == 2
            logits.append(class_score)

            metric_score = class_score / num_support
            metric_loss += metric_score[class_idx == labels].mean()

        logits = torch.cat(logits, dim=1)
        assert logits.size() == (num_query, num_classes)

        class_loss = F.cross_entropy(logits, labels)
        total_loss = class_loss + self.lambda_metric * metric_loss

        self.log("losses/class_loss", class_loss, on_step=True)
        self.log("losses/metric_loss", metric_loss, on_step=True)
        self.log("losses/total_loss", total_loss, on_step=True)

        return total_loss


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


class RelationNetClassifier_Protonet1(pl.LightningModule, FewshotClassifier):

    def __init__(self, simnet_channels=[128, 64, 32], lr=1e-3, weight_decay=1e-4):
        pl.LightningModule.__init__(self)

        self.features = Protonet(in_channels=416, mid_channels=[], out_channels=32)
        self.features.load_state_dict(torch.load("protonet_1.pth"))
        for p in self.features.parameters():
            p.requires_grad = False

        self.simnet = Simnet(in_channels=32, channels=simnet_channels)

        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluators = nn.ModuleDict()

    def forward(self, queries: torch.Tensor, *supports: _T.List[torch.Tensor]):
        assert queries.dim() == 2

        num_query, num_dim = queries.size()
        num_classes = len(supports)

        assert all([class_supports.dim() == 2 for class_supports in supports])
        assert all([class_supports.size(1) == num_dim for class_supports in supports])

        queries_features = self.features(queries)

        scores = []
        for class_supports in supports:
            class_score = 0
            num_support = class_supports.size(0)
            for i in range(num_support):
                item = class_supports[i]
                item = item.unsqueeze(0)

                item_features = self.features(item)
                item_features = item_features.repeat(num_query, 1)

                item_score = self.simnet(queries_features, item_features)
                if item_score.dim() == 1:
                    item_score.unsqueeze_(1)

                assert item_score.dim() == 2
                class_score += item_score

            assert class_score.dim() == 2
            scores.append(class_score)

        scores = torch.cat(scores, dim=1)
        assert scores.size() == (num_query, num_classes)
        return scores

    def configure_optimizers(self):
        return optim.Adam(self.simnet.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def validation_step(self, batch: T.List[torch.Tensor], batch_idx: int, dataloader_idx: int):
        queries, labels, *supports = batch
        logits = self(queries, *supports)

        if dataloader_idx not in self.evaluators:
            eval_n_classes = len(supports)
            self.evaluators[f"dl_{dataloader_idx}"] = nn.ModuleDict({
                "accuracy": plmc.Accuracy(),
                "precision": plmc.Precision(num_classes=eval_n_classes),
                "recall": plmc.Recall(num_classes=eval_n_classes),
                "fbeta": plmc.FBeta(num_classes=eval_n_classes),
                "f1": plmc.F1(num_classes=eval_n_classes),
                "confmat": plmc.ConfusionMatrix(num_classes=eval_n_classes)
            }).to(device=self.device)

        evaluators = self.evaluators[f"dl_{dataloader_idx}"]
        for category, evaluator in evaluators.items():
            self.log(f"metrics/{category}", evaluator(logits, labels))

    def training_step(self, batch: T.List[torch.Tensor], batch_idx: int):
        queries, labels, *supports = batch

        assert queries.dim() == 2

        num_query, num_dim = queries.size()
        num_classes = len(supports)

        assert all([class_supports.dim() == 2 for class_supports in supports])
        assert all([class_supports.size(1) == num_dim for class_supports in supports])

        queries_features = self.features(queries)

        logits = []
        for class_idx, class_supports in enumerate(supports):
            class_score = 0
            # metric_score = 0
            num_support = class_supports.size(0)
            for i in range(num_support):
                item = class_supports[i]
                item = item.unsqueeze(0)

                item_features = self.features(item)
                item_features = item_features.repeat(num_query, 1)

                item_score = self.simnet(queries_features, item_features)
                if item_score.dim() == 1:
                    item_score.unsqueeze_(1)

                assert item_score.dim() == 2
                class_score += item_score

                # metric_score += item_score[labels == class_idx]

            assert class_score.dim() == 2
            logits.append(class_score)

        logits = torch.cat(logits, dim=1)
        assert logits.size() == (num_query, num_classes)

        class_loss = F.cross_entropy(logits, labels)

        self.log("losses/class_loss", class_loss, on_step=True)
        return class_loss
