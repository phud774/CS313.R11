"""Measure Actor cosine similarity with the DGMAE repo's reconstruction model."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from run_figure8 import DGMAE_GRAPHMAE_COMMIT, ROOT, cosine_scores, plot, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgmae-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args()
    checkout = cli.dgmae_repo.resolve()
    if revision(checkout) != DGMAE_GRAPHMAE_COMMIT:
        parser.error(f"Expected DGMAE checkout at {DGMAE_GRAPHMAE_COMMIT}")
    sys.path.insert(0, str(checkout))

    from dgl.data import ActorDataset
    from graphmae.datasets.data_util import load_dataset
    from graphmae.models import build_model
    from graphmae.utils import create_optimizer, load_best_configs, set_random_seed

    model_path = Path(build_model.__code__.co_filename).resolve()
    if not model_path.is_relative_to(checkout):
        raise ImportError(f"DGMAE model resolved to {model_path}")
    print(f"DGMAE model: {model_path}", flush=True)
    # Defaults from the authors' build_args for fields absent from configs.yml.
    config = load_best_configs(
        SimpleNamespace(dataset="act", num_out_heads=1, negative_slope=0.2, concat_hidden=False),
        checkout / "configs.yml",
    )
    set_random_seed(0)
    graph, (config.num_features, _) = load_dataset(config.dataset)
    config.num_nodes = graph.num_nodes()
    graph = graph.to(cli.device)
    x = graph.ndata["feat"]
    model = build_model(config).to(cli.device)
    optimizer = create_optimizer(config.optimizer, model, config.lr, config.weight_decay)
    epochs = 1 if cli.smoke else config.max_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (1 + np.cos(epoch * np.pi / epochs)) * 0.5,
    )
    model.train()
    for epoch in range(1, epochs + 1):
        mask = torch.randperm(config.num_nodes)[:int(config.mask_rate * config.num_nodes)].to(cli.device)
        _, loss, _, _, _ = model(graph, x, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite feature reconstruction loss at epoch {epoch}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if config.scheduler:
            scheduler.step()
        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            print(f"GraphMAE {epoch}/{epochs}: {loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        embedding = model.embed(graph, x).cpu()
    raw_graph = ActorDataset()[0]
    raw = raw_graph.ndata["feat"].float()
    labels = graph.ndata["label"].cpu()
    if not torch.equal(labels, raw_graph.ndata["label"]):
        raise ValueError("Raw and processed Actor node labels do not match")
    source, target = (edge.cpu() for edge in graph.edges())
    different = labels[source] != labels[target]
    low = torch.minimum(source[different], target[different])
    high = torch.maximum(source[different], target[different])
    keys = torch.unique(high * config.num_nodes + low, sorted=True)
    pairs = torch.stack((keys % config.num_nodes, keys // config.num_nodes))

    cli.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cli.output / "actor.npz", pairs=pairs.numpy(),
        raw=cosine_scores(raw, pairs), graphmae=cosine_scores(embedding, pairs),
    )
    metadata = {
        "figure": "DGMAE Figure 8(c)",
        "commit": revision(ROOT), "dgmae_commit": revision(checkout),
        "configuration": "configs.yml:act", "objective": "feature reconstruction only",
        "seed": 0, "epochs": epochs, "smoke": cli.smoke,
        "device": cli.device, "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name() if cli.device == "cuda" else None,
        "num_edges": graph.num_edges(), "pair_count": pairs.shape[1],
        "pair_policy": "all unique undirected cross-label edges",
        "model_config": vars(config),
    }
    (cli.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot(cli.output, ("actor",))
    print(f"Saved Figure 8 artifacts to {cli.output}", flush=True)


if __name__ == "__main__":
    main()
