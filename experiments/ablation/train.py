"""Pretrain the upstream DGMAE backbone with one ablation at a time."""

import numpy as np
import torch

from graphmae.models import build_model
from graphmae.models.loss_func import sce_loss
from graphmae.utils import create_optimizer


def sparse_neighbors(row, col, values, features):
    nodes = features.shape[0]
    adjacency = torch.sparse_coo_tensor(torch.stack((row, col)), values, (nodes, nodes)).coalesce()
    return torch.sparse.mm(adjacency, features)


def paper_target(features, edges, inv_degree, selected=None):
    # Rows are the GAT receivers. Self-loops belong to the backbone only.
    src, dst = edges
    keep = src != dst
    if selected is not None:
        keep &= selected
    src, dst = src[keep], dst[keep]
    weights = inv_degree[dst] * inv_degree[src]
    neighbors = sparse_neighbors(dst, src, weights, features)
    if selected is None:
        return features - neighbors  # Eq. 5
    degree = weights.new_zeros(features.shape[0]).index_add_(0, dst, weights)
    return degree[:, None] * features - neighbors  # Eq. 9


def upstream_target(features, edges, hops):
    row, col = edges
    degree = torch.bincount(row, minlength=features.shape[0]).to(features.dtype)
    inv = degree.clamp_min(1).rsqrt() * (degree > 0)
    weights = inv[row] * inv[col]
    target = features
    for _ in range(hops):
        target = target - sparse_neighbors(row, col, weights, target)
    return target


def paper_reconstruction(model, graph, features, mask, capture_attention):
    masked_features = features.masked_fill(mask[:, None], 0)
    if capture_attention:
        attention = []
        last_dropout = model.encoder.gat_layers[-1].attn_drop
        handle = last_dropout.register_forward_pre_hook(lambda _, inputs: attention.append(inputs[0].detach()))
        try:
            encoded, _ = model.encoder(graph, masked_features)
        finally:
            handle.remove()
        coefficients = attention[0]
    else:
        encoded, _ = model.encoder(graph, masked_features)
        coefficients = None
    decoded, _ = model.decoder(graph, model.encoder_to_decoder(encoded))
    return decoded, coefficients


def pretrain(config, graph, variant, protocol):
    features = graph.ndata["feat"]
    device = features.device
    model = build_model(config).to(device)
    optimizer = create_optimizer(config.optimizer, model, config.lr, config.weight_decay)
    scheduler = None
    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: (1 + np.cos(epoch * np.pi / config.max_epoch)) * 0.5
        )
    use_feature = variant != "no-feature"
    use_discrepancy = variant != "no-discrepancy"
    use_selection = use_discrepancy and variant != "no-selection"
    edges = torch.stack(graph.edges(order="eid"))
    inv_degree = (graph.in_degrees() - 1).clamp_min(1).to(features.dtype).rsqrt()
    paper = protocol == "paper"
    if use_discrepancy and not use_selection:
        fixed_target = (
            paper_target(features, edges, inv_degree)
            if paper
            else upstream_target(features, edges, config.hop)
        )

    model.train()
    last_losses = None
    for epoch in range(config.max_epoch):
        if paper:
            mask = torch.rand(graph.num_nodes(), device=device) < config.mask_rate
            reconstruction, attention = paper_reconstruction(model, graph, features, mask, use_selection)
            feature_loss = model.criterion(reconstruction[mask], features[mask]) if use_feature else None
        else:
            permutation = torch.randperm(graph.num_nodes()).to(device)
            masked_nodes = permutation[: int(config.mask_rate * graph.num_nodes())]
            mask = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
            mask[masked_nodes] = True
            if use_discrepancy:
                unmasked, _ = model.encoder(graph, features)
            reconstruction, original_loss, _, attention, _ = model(graph, features, masked_nodes)
            feature_loss = original_loss if use_feature else None

        discrepancy_loss = None
        if use_discrepancy:
            if paper:
                unmasked, _ = model.encoder(graph, features)
            target = fixed_target if not use_selection else None
            if use_selection:
                weights = attention.detach().mean(dim=1).reshape(-1)
                if paper:
                    probability = ((1 - weights) * config.pc).clamp(0, config.pt)
                    target = paper_target(features, edges, inv_degree, torch.bernoulli(probability).bool())
                else:
                    weights = 1 - weights.sigmoid()
                    probability = (weights / weights.mean() * config.pc).clamp_max(config.pt)
                    target = upstream_target(features, edges[:, torch.bernoulli(probability).bool()], config.hop)
            discrepancy = model.mlp(unmasked) - reconstruction
            discrepancy_loss = sce_loss(discrepancy[~mask], target[~mask], config.gamma2)

        # Keep the original coefficient for each retained term of Eq. 13.
        loss = 0
        if use_feature:
            loss = (1 - config.discrepancy_weight) * feature_loss
        if use_discrepancy:
            loss = loss + config.discrepancy_weight * discrepancy_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1} for {variant}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch + 1 == config.max_epoch:
            last_losses = {
                "total": float(loss.detach()),
                "feature": float(feature_loss.detach()) if use_feature else None,
                "discrepancy": float(discrepancy_loss.detach()) if use_discrepancy else None,
            }
            print(f"  pretrain {epoch + 1}/{config.max_epoch} {last_losses}", flush=True)

    model.eval()
    with torch.no_grad():
        return model.embed(graph, features), last_losses
