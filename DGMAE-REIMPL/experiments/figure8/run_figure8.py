"""Train Figure 8 encoders and plot cosine similarity on cross-label edges."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dgmae.data import _load_raw, load_node_dataset
from dgmae.utils import set_seed
from train_node import build_model, pretrain

DATASETS = ("roman", "squirrel", "actor")
MODELS = ("graphmae", "greet", "dgmae")
GRAPHMAE_COMMIT = "b14f080c919257b495e3cb6474286d5252d6a635"
GREET_COMMIT = "8bcc94090fe5f343aaeedcda981c5344d496a7f1"


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def train_graphmae(x, graph, cfg, seed):
    import dgl
    from graphmae.models import build_model as build_graphmae
    from graphmae.utils import create_optimizer, set_random_seed

    set_random_seed(seed)
    src, dst = graph["edge_index"]
    g = dgl.graph((src, dst), num_nodes=len(x), device=x.device)
    args = dict(
        num_features=x.size(1), num_hidden=cfg["num_hidden"], num_heads=cfg["heads"],
        num_out_heads=1, num_layers=cfg["num_layers"], residual=cfg.get("residual", False),
        norm=cfg.get("norm"), mask_rate=cfg["mask_rate"], replace_rate=cfg["replace_rate"],
        drop_edge_rate=0.0, in_drop=0.2, attn_drop=0.1, negative_slope=0.2,
        alpha_l=cfg["gamma1"], encoder="gat", decoder="gat", activation="prelu",
        loss_fn="sce", concat_hidden=False, optimizer="adam", lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], max_epoch=cfg["max_epoch"],
    )
    model = build_graphmae(SimpleNamespace(**args)).to(x.device)
    optimizer = create_optimizer(args["optimizer"], model, args["lr"], args["weight_decay"])
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (1 + math.cos(epoch * math.pi / args["max_epoch"])) * 0.5,
    )
    model.train()
    for epoch in range(1, args["max_epoch"] + 1):
        loss, _ = model(g, x)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite GraphMAE loss at epoch {epoch}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if epoch == 1 or epoch % 100 == 0 or epoch == args["max_epoch"]:
            print(f"GraphMAE {epoch}/{args['max_epoch']}: {loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        return model.embed(g, x).cpu(), args


def cosine_scores(embedding, pairs):
    if not torch.isfinite(embedding).all():
        raise ValueError("Embedding contains non-finite values")
    vectors = F.normalize(embedding.detach().cpu(), dim=1)
    return torch.cat([
        (vectors[left] * vectors[right]).sum(dim=1)
        for left, right in pairs.split(4096, dim=1)
    ]).clamp(-1, 1).numpy()


def plot(output, datasets):
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    grid = np.linspace(-1, 1, 401)
    labels = {"raw": ("Raw feature", "#d5535e"), "graphmae": ("GraphMAE", "#5075b6"),
              "greet": ("GREET", "#5c9b73"), "dgmae": ("DGMAE", "#df874c")}
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 3.4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        with np.load(output / f"{dataset}.npz") as scores:
            for key, (label, color) in labels.items():
                if key not in scores:
                    continue
                values = scores[key]
                if np.ptp(values) < 1e-7:
                    ax.axvline(values[0], label=label, color=color)
                    continue
                density = gaussian_kde(values)(grid)
                ax.plot(grid, density, color=color, label=label, linewidth=1.3)
                ax.fill_between(grid, density, color=color, alpha=0.27)
        ax.set(title=dataset.capitalize(), xlim=(-1, 1), ylim=(0, None),
               xlabel="Cosine similarity", ylabel="Density")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figure8.png", dpi=180)
    fig.savefig(output / "figure8.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphmae", type=Path)
    p.add_argument("--greet", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--graphmae-features", choices=("standard", "raw"), default="standard",
                   help="Use the dataset preprocessing or the raw features for GraphMAE")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--smoke", action="store_true", help="Train each selected encoder for one epoch")
    args = p.parse_args()
    for model, checkout, commit in (("graphmae", args.graphmae, GRAPHMAE_COMMIT),
                                    ("greet", args.greet, GREET_COMMIT)):
        if model in args.models:
            if checkout is None or revision(checkout) != commit:
                p.error(f"{model} requires the pinned checkout at {commit}")
            sys.path.append(str(checkout.resolve()))
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    if "greet" in args.models and args.device != "cuda":
        p.error("The official GREET trainer requires CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/node.yml").read_text())
    metadata = {
        "paper": "doi:10.1145/3711896.3736912, Figure 8",
        "commit": revision(ROOT), "graphmae_commit": GRAPHMAE_COMMIT, "greet_commit": GREET_COMMIT,
        "seed": args.seed, "smoke": args.smoke, "device": args.device,
        "torch": str(torch.__version__), "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "pair_policy": "all unique undirected, non-self-loop edges whose endpoint labels differ",
        "datasets": {},
    }
    for dataset in args.datasets:
        x, labels, graph, _, info = load_node_dataset(dataset, str(args.data), args.device)
        raw = _load_raw(dataset, str(args.data))[0].x.float()
        edges = graph["edge_index"][:, ~graph["self_loop"]].cpu()
        left, right = edges
        labels = labels.cpu()
        pairs = edges[:, (left < right) & (labels[left] != labels[right])]
        if pairs.shape[1] < 2:
            raise ValueError(f"{dataset} has too few cross-label edges")
        cfg = dict(config[dataset], selection="inv_attn", hop=1)
        if args.smoke:
            cfg["max_epoch"] = 1
        info.update(pair_count=pairs.size(1), configs={},
                    feature_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
                    edge_sha256=hashlib.sha256(edges.numpy().tobytes()).hexdigest(),
                    label_sha256=hashlib.sha256(labels.numpy().tobytes()).hexdigest())
        metadata["datasets"][dataset] = info
        scores = {"pairs": pairs.numpy(), "raw": cosine_scores(raw, pairs)}
        for name in args.models:
            print(f"{dataset} / {name} / seed {args.seed}", flush=True)
            if name == "dgmae":
                set_seed(args.seed)
                model = build_model(x.size(1), cfg).to(args.device)
                history = pretrain(model, x, graph, cfg, log_every=100)
                if not math.isfinite(history[-1]["loss"]):
                    raise FloatingPointError("Non-finite DGMAE loss")
                model.eval()
                embedding, settings = model.embed(x, graph).cpu(), cfg
                del model
            elif name == "graphmae":
                graphmae_x = raw.to(args.device) if args.graphmae_features == "raw" else x
                embedding, settings = train_graphmae(graphmae_x, graph, cfg, args.seed)
                settings["input_features"] = args.graphmae_features
            else:
                from greet_baseline import train_greet
                embedding, settings = train_greet(dataset, raw, edges, args.seed, args.smoke)
            if len(embedding) != len(raw):
                raise ValueError(f"{name} returned embeddings with the wrong node count")
            scores[name] = cosine_scores(embedding, pairs)
            info["configs"][name] = settings
        np.savez_compressed(args.output / f"{dataset}.npz", **scores)
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot(args.output, args.datasets)
    print(f"Saved Figure 8 artifacts to {args.output}", flush=True)


if __name__ == "__main__":
    main()
