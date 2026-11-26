"""Pretrain the upstream DGMAE backbone with one ablation at a time."""

import numpy as np
import torch

from graphmae.models import build_model
from graphmae.models.loss_func import sce_loss
from graphmae.utils import create_optimizer


def sparse_adjacency(row, col, values, nodes):
    return torch.sparse_coo_tensor(torch.stack((row, col)), values, (nodes, nodes)).coalesce()


def paper_targets(features, edges, inv_degree, selected=None, *, include_two_hop=False):
    # Rows are the GAT receivers. Self-loops belong to the backbone only.
    src, dst = edges
    keep = src != dst
    if selected is not None:
        keep &= selected
    src, dst = src[keep], dst[keep]
    weights = inv_degree[dst] * inv_degree[src]
    adjacency = sparse_adjacency(dst, src, weights, features.shape[0])
    neighbors = torch.sparse.mm(adjacency, features)
    if selected is None:
        diagonal = 1  # Eq. 5
    else:
        diagonal = weights.new_zeros(features.shape[0]).index_add_(0, dst, weights)[:, None]  # Eq. 9
    one_hop = diagonal * features - neighbors
    two_hop = None
    if include_two_hop:
        two_hop = diagonal * one_hop - torch.sparse.mm(adjacency, one_hop)
    return one_hop, two_hop


def upstream_target(features, edges, hops):
    row, col = edges
    degree = torch.bincount(row, minlength=features.shape[0]).to(features.dtype)
    inv = degree.clamp_min(1).rsqrt() * (degree > 0)
    weights = inv[row] * inv[col]
    adjacency = sparse_adjacency(row, col, weights, features.shape[0])
    target = features
    for _ in range(hops):
        target = target - torch.sparse.mm(adjacency, target)
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


def pretrain(config, graph, variant, protocol, probe_epochs=()):
    features = graph.ndata["feat"]
    device = features.device
    model = build_model(config).to(device)
    optimizer = create_optimizer(config.optimizer, model, config.lr, config.weight_decay)
    scheduler = None
    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: (1 + np.cos(epoch * np.pi / config.max_epoch)) * 0.5
        )
    use_feature = variant != "no-feature" and config.discrepancy_weight < 1
    use_discrepancy = variant != "no-discrepancy" and config.discrepancy_weight > 0
    use_selection = use_discrepancy and variant != "no-selection"
    edges = torch.stack(graph.edges(order="eid"))
    inv_degree = (graph.in_degrees() - 1).clamp_min(1).to(features.dtype).rsqrt()
    paper = protocol == "paper"
    include_two_hop = paper and config.discrepancy_neighborhood == "one-hop-two-hop"
    discrepancy_coefficient = 1.0 if paper and variant == "no-feature" else config.discrepancy_weight
    if use_discrepancy and not use_selection:
        fixed_targets = (
            paper_targets(features, edges, inv_degree, include_two_hop=include_two_hop)
            if paper
            else (upstream_target(features, edges, config.hop), None)
        )

    model.train()
    last_losses = None
    snapshots = {}
    probe_epochs = set(probe_epochs)
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
            if not use_selection:
                target, two_hop_target = fixed_targets
            else:
                weights = attention.detach().mean(dim=1).reshape(-1)
                if paper:
                    probability = ((1 - weights) * config.pc).clamp(0, config.pt)
                    selected = torch.bernoulli(probability).bool()
                    target, two_hop_target = paper_targets(
                        features, edges, inv_degree, selected, include_two_hop=include_two_hop,
                    )
                else:
                    weights = 1 - weights.sigmoid()
                    probability = (weights / weights.mean() * config.pc).clamp_max(config.pt)
                    target = upstream_target(features, edges[:, torch.bernoulli(probability).bool()], config.hop)
                    two_hop_target = None
            discrepancy = (model.mlp(unmasked) - reconstruction)[~mask]
            discrepancy_loss = sce_loss(discrepancy, target[~mask], config.gamma2)
            if two_hop_target is not None:
                two_hop_loss = sce_loss(discrepancy, two_hop_target[~mask], config.gamma2)
                discrepancy_loss = (discrepancy_loss + two_hop_loss) / 2

        # Use the configured feature coefficient for every variant that retains it.
        loss = 0
        if use_feature:
            loss = (1 - config.discrepancy_weight) * feature_loss
        if use_discrepancy:
            loss = loss + discrepancy_coefficient * discrepancy_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1} for {variant}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        epoch_number = epoch + 1
        if epoch_number % 100 == 0 or epoch_number in (1, config.max_epoch) or epoch_number in probe_epochs:
            last_losses = {
                "total": float(loss.detach()),
                "feature": float(feature_loss.detach()) if use_feature else None,
                "discrepancy": float(discrepancy_loss.detach()) if use_discrepancy else None,
            }
            print(f"  pretrain {epoch + 1}/{config.max_epoch} {last_losses}", flush=True)
        if epoch_number in probe_epochs and epoch_number != config.max_epoch:
            model.eval()
            with torch.no_grad():
                snapshots[epoch_number] = (model.embed(graph, features).cpu(), last_losses)
            model.train()

    model.eval()
    with torch.no_grad():
        return model.embed(graph, features), last_losses, snapshots
