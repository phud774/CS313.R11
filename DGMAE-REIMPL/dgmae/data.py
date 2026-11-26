"""Nạp dataset node-level qua PyG và tiền xử lý.

Tiền xử lý đặc trưng (paper không nêu; theo đúng code tác giả để so sánh được):
  - cora/citeseer/pubmed : chuẩn hóa theo hàng (bản DGL tác giả dùng đã chuẩn hóa sẵn)
  - chameleon/squirrel/actor : StandardScaler
  - roman : PairNorm (PN-SCS, scale=10)
  - texas/cornell/wisconsin : giữ nguyên (bag-of-words nhị phân)
Đồ thị: chuyển vô hướng, bỏ cạnh trùng, thêm self-loop cho GAT.
Bậc d_i trong Eq. 4/9 tính trên A KHÔNG có self-loop (đúng định nghĩa Mục 3).
Split:
  - Planetoid: split public (140/500/1000 với Cora), 1 split
  - Texas/Cornell/Wisconsin/Chameleon/Squirrel/Actor: 10 split Geom-GCN (48/32/20)
  - Roman-empire: 10 split của Platonov et al. (50/25/25)
"""
import torch
import torch_geometric.datasets as D
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, degree

HOMOPHILIC = ("cora", "citeseer", "pubmed")
HETEROPHILIC = ("texas", "cornell", "wisconsin", "chameleon", "squirrel", "actor", "roman")


def _load_raw(name, root):
    if name == "cora":
        return D.Planetoid(root, "Cora")
    if name == "citeseer":
        return D.Planetoid(root, "CiteSeer")
    if name == "pubmed":
        return D.Planetoid(root, "PubMed")
    if name in ("texas", "cornell", "wisconsin"):
        return D.WebKB(root, name.capitalize())
    if name in ("chameleon", "squirrel"):
        return D.WikipediaNetwork(root, name)
    if name == "actor":
        return D.Actor(f"{root}/actor")
    if name == "roman":
        return D.HeterophilousGraphDataset(root, "Roman-empire")
    raise ValueError(f"Unknown dataset {name}")


def _standardize(x):
    return (x - x.mean(0)) / x.std(0, unbiased=False).clamp_min(1e-12)


def _pairnorm_scs(x, scale=10.0):
    return scale * x / (1e-6 + x.pow(2).sum(1, keepdim=True)).sqrt() - x.mean(0)


def load_node_dataset(name, root="data", device="cpu"):
    ds = _load_raw(name, root)
    d = ds[0]
    x = d.x.float()
    if name in HOMOPHILIC:
        x = x / x.sum(1, keepdim=True).clamp_min(1)
    elif name in ("chameleon", "squirrel", "actor"):
        x = _standardize(x)
    elif name == "roman":
        x = _pairnorm_scs(x)

    n = x.size(0)
    ei, _ = remove_self_loops(d.edge_index)
    ei = to_undirected(ei, num_nodes=n)                   # cũng loại bỏ cạnh trùng
    deg = degree(ei[1], n)                                 # d_i không tính self-loop
    ei_sl, _ = add_self_loops(ei, num_nodes=n)
    ei_sl = ei_sl[:, torch.argsort(ei_sl[1] * n + ei_sl[0])]  # sắp theo (đích, nguồn) -> ma trận thưa đã coalesce
    self_loop = ei_sl[0] == ei_sl[1]

    train, val, test = d.train_mask, d.val_mask, d.test_mask
    if train.dim() == 1:
        train, val, test = train[:, None], val[:, None], test[:, None]

    graph = {
        "edge_index": ei_sl.to(device),
        "self_loop": self_loop.to(device),
        "deg": deg.to(device),
        "deg_inv_sqrt": deg.pow(-0.5).masked_fill(deg == 0, 0).to(device),
    }
    y = d.y.long()
    h_edge = (y[ei[0]] == y[ei[1]]).float().mean().item()
    info = {"name": name, "num_nodes": n, "num_edges": ei.size(1) // 2, "num_features": x.size(1),
            "num_classes": int(y.max()) + 1, "num_splits": train.size(1), "edge_homophily": h_edge}
    return x.to(device), y.to(device), graph, (train.to(device), val.to(device), test.to(device)), info
