import torch
import torch.nn

import utils_data as D


def simnet_count(queries: torch.Tensor, supports: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module):
    assert queries.size(0) == labels.size(0)
    assert queries.shape[1:] == supports.shape[1:]

    scores = []
    for support in supports:
        support = support.unsqueeze(0).repeat(queries.size(0))
        score = model(queries, support).mean(dim=1, keepdim=True)
        scores.append(score)

    scores = torch.cat(scores, dim=1)
    preds = scores.max(dim=1)[1]

    return (preds == labels).sum().item()


def simnet(datasets, model, n_query=1000, n_support=10):
    supports = []
    for dataset in datasets:
        dataloader = D.DataLoader(dataset, shuffle=True, batch_size=n_support)
        support = next(iter(dataloader))
        support = support.mean(dim=0, keepdim=True)
        supports.append(support)
    supports = torch.cat(supports, dim=0)

    correct, total = 0, 0
    for label, dataset in enumerate(datasets):
        dataloader = D.DataLoader(dataset, shuffle=True, batch_size=n_query, model=model)
        query = next(iter(dataloader))

        total += n_query
        correct += simnet_count(query, supports, torch.tensor([label] * n_query, device=query.device))

    return correct / total
