"""Adapter for the official GREET trainer on the shared Figure 8 graphs."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch.utils.checkpoint import checkpoint


# Actor and Squirrel overrides follow GREET scripts/run_{actor,squirrel}.sh.
PUBLISHED = {
    "actor": dict(epochs=1000, sparse=1, nlayers_proj=2, dropedge_rate_2=0.8, margin_hom=0.1),
    "squirrel": dict(epochs=1000, nlayers_proj=2, k=0, maskfeat_rate_2=0.1,
                     dropedge_rate_1=0.1, dropedge_rate_2=0.8, margin_hom=0.1, margin_het=0.3),
    "roman": dict(sparse=1),
}


def full_batch_loss(anchor, sample, neighbors, temperature=0.2, block_size=256):
    """GREET full-batch InfoNCE, with recomputation to bound memory on Roman."""
    anchor, sample = F.normalize(anchor), F.normalize(sample)
    count = len(anchor)

    def block_loss(a, b, positives):
        similarity = (a @ b.T) / temperature
        negative_sum = similarity.masked_fill(positives, -torch.inf).logsumexp(dim=1)
        positive_mean = similarity.masked_fill(~positives, 0).sum(dim=1) / positives.sum(dim=1)
        return (negative_sum - positive_mean).sum()

    total = anchor.new_zeros(())
    for start in range(0, count, block_size):
        end = min(start + block_size, count)
        positives = torch.from_numpy(neighbors[start:end].toarray()).to(anchor.device, torch.bool)
        rows = torch.arange(end - start, device=anchor.device)
        positives[rows, rows + start] = True
        total = total + checkpoint(block_loss, anchor[start:end], sample, positives, use_reentrant=False)
    return total / count


def train_greet(dataset, raw, edges, seed, smoke):
    # The upstream implementation uses the default CUDA device.
    from data_loader import get_structural_encoding
    import main as upstream
    from model import Edge_Discriminator, GCL
    from torch_geometric.data import Data
    from torch_geometric.transforms import NormalizeFeatures

    options = dict(epochs=400, sparse=0, nlayers_enc=2, nlayers_proj=1, emb_dim=128, proj_dim=128,
                   dropout=0.5, cl_batch_size=0, alpha=0.1, k=20, lr_gcl=0.001, lr_disc=0.001,
                   w_decay=0.0, cl_rounds=2, maskfeat_rate_1=0.1, maskfeat_rate_2=0.5,
                   dropedge_rate_1=0.5, dropedge_rate_2=0.1, margin_hom=0.5, margin_het=0.5)
    options.update(PUBLISHED[dataset])
    if smoke:
        options["epochs"] = 1
    cfg = SimpleNamespace(**options)
    upstream.args = cfg
    upstream.setup_seed(seed)
    features = NormalizeFeatures()(Data(x=raw.clone())).x if dataset == "squirrel" else raw
    # Clamping the degree of isolated nodes to zero is implicit in the sparse
    # walks because those nodes have no incident edges.
    print(f"{dataset} GREET: computing structural encodings", flush=True)
    structural = get_structural_encoding(edges, len(features)).cuda()

    class RomanGCL(GCL):
        def batch_nce_loss(self, z1, z2, temperature=0.2, pos_mask=None, neg_mask=None):
            return (full_batch_loss(z1, z2, self.neighbors, temperature)
                    + full_batch_loss(z2, z1, self.neighbors, temperature)) / 2

    model_class = RomanGCL if dataset == "roman" else GCL
    model = model_class(cfg.nlayers_enc, cfg.nlayers_proj, features.size(1), cfg.emb_dim,
                        cfg.proj_dim, cfg.dropout, cfg.sparse, cfg.cl_batch_size).cuda()
    if dataset == "roman":
        model.neighbors = kneighbors_graph(features.numpy(), cfg.k, metric="cosine")
    else:
        model.set_mask_knn(features, cfg.k, dataset)
    discriminator = Edge_Discriminator(len(features), features.size(1) + structural.size(1),
                                       cfg.alpha, cfg.sparse).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_gcl, weight_decay=cfg.w_decay)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_disc, weight_decay=cfg.w_decay)
    features, edges = features.cuda(), edges.cuda()
    for epoch in range(1, cfg.epochs + 1):
        for _ in range(cfg.cl_rounds):
            contrastive = upstream.train_cl(model, discriminator, opt, features, structural, edges)
        ranking = upstream.train_discriminator(model, discriminator, disc_opt, features, structural, edges, cfg)
        if not torch.isfinite(torch.tensor([contrastive, ranking])).all():
            raise FloatingPointError(f"Non-finite GREET loss at epoch {epoch}")
        if epoch == 1 or epoch % 50 == 0 or epoch == cfg.epochs:
            print(f"{dataset} GREET {epoch}/{cfg.epochs}: {contrastive:.4f}, {ranking:.4f}", flush=True)
    model.eval()
    discriminator.eval()
    with torch.no_grad():
        a1, a2, *_ = discriminator(torch.cat((features, structural), dim=1), edges)
        return model.get_embedding(features, a1, a2).cpu(), options
