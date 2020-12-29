import torch
import typing as T
import utils.data as data

import pytorch_lightning.metrics as plmc

import dataloader
import model


def select_batch(dataset: data.Dataset, size: int, generator: torch.Generator = None):
    dataloader = data.DataLoader(dataset, batch_size=size, shuffle=True, generator=generator)
    return next(iter(dataloader))


def build_dataloader(datasets: T.Dict[str, data.Dataset], n_support: int, n_queries: int, generator: torch.Generator = None):

    dslist = list(datasets.values())

    queries = dslist[0]
    labels = [0] * len(dslist[0])

    for i in range(1, len(dslist)):
        queries += dslist[i]
        labels += [i] * len(dslist[i])

    # Select supports
    # Will be constant for all queries
    supports = []
    for dataset in dslist:
        support = select_batch(dataset, size=n_support, generator=generator)
        supports.append(support)

    def collate_fn(batch):
        queries = []
        labels = []
        for row in batch:
            query, label = row
            queries.append(query)
            labels.append(label)
        queries = torch.stack(queries, dim=0)
        labels = torch.tensor(labels, device=queries.device)
        return queries, labels, *supports

    dataset = data.dzip(queries, labels)
    return data.DataLoader(dataset, batch_size=n_queries, collate_fn=collate_fn, shuffle=True, generator=generator)


load = dataloader.iscxvpn2016(pcap_dir="D://Datasets/ISCXVPN2016/", h5_dir="D://Datasets/packets-15k/", as_bit=True)
datasets = {
    "a": load("youtube"),
    "b": load("youtube"),
    "c": load("youtube"),
    "d": load("youtube")
}
confmat = plmc.ConfusionMatrix(num_classes=4)

network = model.ProtonetClassifier(416, 10)
dl = build_dataloader(datasets, n_support=50, n_queries=10)

step = 0
for queries, labels, *supports in dl:

    # print(a.size())
    # print(b.size())
    # print(len(supports))
    # for i, s in enumerate(supports):
    #     print(f"support {i}", s.size())

    logits = network(queries, *supports)

    print(confmat(logits, labels))
    step += 1

    if step > 10:
        break
