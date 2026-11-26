"""DGL dataset loading and Table 7 evaluation masks."""

import hashlib
from pathlib import Path

import dgl
import torch
from dgl.data import AmazonCoBuyPhotoDataset, CiteseerGraphDataset, CoraGraphDataset, RomanEmpireDataset
from dgl.data.utils import download

from graphmae.datasets.data_util import PairNorm, load_dataset
from protocol import CITESEER_SOURCE_COMMIT, DATASETS, ROMAN_SHA256, ROMAN_SOURCE_COMMIT


class PinnedRomanDataset(RomanEmpireDataset):
    def download(self):
        url = (
            "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/"
            f"{ROMAN_SOURCE_COMMIT}/data/roman_empire.npz"
        )
        download(url, path=str(self.source_file))

    @property
    def source_file(self):
        return Path(self.raw_path) / "roman_empire.npz"

    def process(self):
        if hashlib.sha256(self.source_file.read_bytes()).hexdigest() != ROMAN_SHA256:
            raise ValueError("Unexpected Roman-empire dataset SHA-256")
        super().process()


class PinnedCiteseerDataset(CiteseerGraphDataset):
    def download(self):
        source = f"https://raw.githubusercontent.com/kimiyoung/planetoid/{CITESEER_SOURCE_COMMIT}/data"
        for suffix in ("x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"):
            filename = f"ind.citeseer.{suffix}"
            download(f"{source}/{filename}", path=str(Path(self.raw_path) / filename))


def load_graph(dataset):
    if dataset == "roman":
        data = PinnedRomanDataset()
        graph = data[0]
        graph.ndata["feat"] = PairNorm("PN-SCS")(graph.ndata["feat"])
        graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
        classes = data.num_classes
    elif dataset == "cora":
        data = CoraGraphDataset()
        graph = dgl.add_self_loop(dgl.remove_self_loop(data[0]))
        classes = data.num_classes
    elif dataset in ("photo", "citeseer"):
        data = AmazonCoBuyPhotoDataset() if dataset == "photo" else PinnedCiteseerDataset()
        graph = dgl.add_self_loop(dgl.remove_self_loop(data[0]))
        classes = data.num_classes
    else:
        graph, (_, classes) = load_dataset(DATASETS[dataset])
    return graph, classes


def random_masks(nodes, seed, train_fraction, validation_fraction):
    permutation = torch.randperm(nodes, generator=torch.Generator().manual_seed(seed))
    train_end = int(nodes * train_fraction)
    validation_end = int(nodes * (train_fraction + validation_fraction))
    masks = []
    for indices in (permutation[:train_end], permutation[train_end:validation_end], permutation[validation_end:]):
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[indices] = True
        masks.append(mask)
    return tuple(masks)


def get_splits(graph, dataset, protocol, roman_split, seed, smoke):
    nodes = graph.num_nodes()
    if dataset == "photo":
        return [("seeded-10-10-80", random_masks(nodes, seed, 0.10, 0.10))]
    if dataset == "roman" and roman_split == "paper-ratio":
        return [("seeded-48-32-20", random_masks(nodes, seed, 0.48, 0.32))]
    masks = tuple(graph.ndata[name].bool() for name in ("train_mask", "val_mask", "test_mask"))
    if masks[0].ndim == 1:
        return [("public", masks)]
    if protocol == "upstream" and not smoke:
        split_ids = range(masks[0].shape[1])
    else:
        split_ids = (seed % masks[0].shape[1],)
    return [(f"public-{index}", tuple(mask[:, index] for mask in masks)) for index in split_ids]
