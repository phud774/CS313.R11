"""Run the four DGMAE ablations, then probe frozen embeddings."""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from types import SimpleNamespace

from protocol import DATASETS, DATA_SOURCES, PAPER_ACCURACY, PAPER_SETTINGS, UPSTREAM_COMMIT, VARIANTS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True, help="Pinned authors' DGMAE checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=("paper", "upstream"), help="Full-run protocol")
    parser.add_argument("--roman-split", choices=("public", "paper-ratio"))
    parser.add_argument(
        "--discrepancy-neighborhood", choices=("one-hop", "one-hop-two-hop"),
        help="Paper protocol only. Defaults to one-hop.",
    )
    parser.add_argument("--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variant", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(10), metavar="0–9")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="One pretraining and two probe epochs")
    parser.add_argument("--probe-epochs", nargs="+", type=int, default=[],
                        help="Compare frozen embeddings at these epochs. A later epoch extends the schedule.")
    args = parser.parse_args()
    if not args.smoke and args.protocol is None:
        parser.error("choose --protocol explicitly for full runs")
    if args.protocol == "paper" and "roman" in args.dataset and not (args.roman_split or args.smoke):
        parser.error("choose --roman-split explicitly for full paper runs on Roman")
    if args.protocol == "upstream" and args.roman_split == "paper-ratio":
        parser.error("upstream runs use Roman's public masks")
    if args.protocol == "upstream" and args.discrepancy_neighborhood is not None:
        parser.error("--discrepancy-neighborhood requires the paper protocol")
    if any(epoch < 1 for epoch in args.probe_epochs):
        parser.error("--probe-epochs must contain positive epochs")
    if args.smoke and any(epoch != 1 for epoch in args.probe_epochs):
        parser.error("--smoke probes only epoch 1")
    args.probe_epochs = sorted(set(args.probe_epochs))
    args.protocol = args.protocol or "paper"
    args.roman_split = args.roman_split or "public"
    args.seeds = list(dict.fromkeys(args.seeds or ([0] if args.smoke else list(range(10)))))
    args.dataset = list(dict.fromkeys(args.dataset))
    args.variant = list(dict.fromkeys(args.variant))
    return args


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def configuration(args, dataset, graph, classes):
    from graphmae.utils import load_best_configs

    config = SimpleNamespace(
        dataset=DATASETS[dataset], num_features=graph.ndata["feat"].shape[1],
        num_nodes=graph.num_nodes(), num_classes=classes,
        num_out_heads=1, residual=False, norm=None, negative_slope=0.2,
        concat_hidden=False, hop=2, pc=0.3, pt=0.7, discrepancy_weight=0.1, gamma2=4,
    )
    load_best_configs(config, args.upstream / "configs.yml")
    if args.protocol == "paper":
        vars(config).update(PAPER_SETTINGS[dataset], alpha_l=3, replace_rate=0)
        config.discrepancy_neighborhood = args.discrepancy_neighborhood or "one-hop"
    if args.smoke:
        config.max_epoch = 1
        config.max_epoch_f = 2
    return config


def summarize(output, dataset, variant, smoke, diagnostic=False):
    records = [json.loads(path.read_text()) for path in sorted(output.glob("seed-*.json"))]
    accuracy = [100 * record["test_accuracy"] for record in records]
    summary = {
        "dataset": dataset, "variant": variant, "smoke": smoke,
        "runs": len(records), "seeds": [record["seed"] for record in records],
        "mean_accuracy_pct": statistics.mean(accuracy),
        "std_accuracy_pct": statistics.pstdev(accuracy),
    }
    if not smoke and not diagnostic:
        paper_mean, paper_std = PAPER_ACCURACY[variant][list(DATASETS).index(dataset)]
        summary.update(paper_mean_pct=paper_mean, paper_std_pct=paper_std, complete=len(records) == 10)
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main():
    args = parse_args()
    args.upstream = args.upstream.resolve()
    revision = git_revision(args.upstream)
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Expected authors' commit {UPSTREAM_COMMIT}, found {revision}")
    sys.path.insert(0, str(args.upstream))
    os.environ.setdefault("DGLBACKEND", "pytorch")
    os.environ.setdefault("DGL_REPO", "https://dgl-data.s3-accelerate.amazonaws.com")

    import dgl
    import torch
    from graphmae.utils import set_random_seed
    from datasets import get_splits, load_graph
    from evaluate import probe
    from train import pretrain

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Choose a GPU runtime or pass --device cpu for a local check.")
    environment = {
        "python": platform.python_version(), "torch": torch.__version__, "dgl": dgl.__version__,
        "cuda": torch.version.cuda, "device": str(device),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    experiment_commit = git_revision(Path(__file__).resolve().parent)
    output = args.output / args.protocol / ("smoke" if args.smoke else "full")
    if args.discrepancy_neighborhood == "one-hop-two-hop":
        output /= args.discrepancy_neighborhood
    if args.probe_epochs:
        output /= "probe-epochs-" + "-".join(map(str, args.probe_epochs))
    print(json.dumps({"experiment_commit": experiment_commit, "upstream_commit": revision, **environment}), flush=True)

    for dataset in args.dataset:
        cpu_graph, classes = load_graph(dataset)
        graph = cpu_graph.to(device)
        config = configuration(args, dataset, graph, classes)
        if args.probe_epochs:
            config.max_epoch = max(config.max_epoch, args.probe_epochs[-1])
        split_policy = args.roman_split if dataset == "roman" else "dataset-default"
        labels = graph.ndata["label"].reshape(-1).long()
        run_settings = {
            "paper": "doi:10.1145/3711896.3736912, Table 4",
            "protocol": args.protocol, "dataset": dataset, "data_source": DATA_SOURCES[dataset],
            "split_policy": split_policy,
            "smoke": args.smoke, "experiment_commit": experiment_commit, "upstream_commit": revision,
            "graph_nodes": graph.num_nodes(), "graph_edges_with_self_loops": graph.num_edges(),
            "configuration": vars(config),
        }
        if args.probe_epochs:
            run_settings["probe_epochs"] = args.probe_epochs
            run_settings["experiment_type"] = "encoder_epoch_diagnostic"
        for variant in args.variant:
            directory = output / dataset / split_policy / variant
            directory.mkdir(parents=True, exist_ok=True)
            settings_path = directory / "settings.json"
            settings = dict(run_settings, variant=variant)
            if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
                raise ValueError(f"Settings differ from earlier runs. Choose a new --output: {directory}")
            save_json(settings_path, settings)
            for seed in args.seeds:
                record_path = directory / f"seed-{seed}.json"
                if record_path.is_file():
                    print(f"Reusing {record_path}", flush=True)
                    continue
                print(f"{dataset} / {variant} / seed {seed}", flush=True)
                set_random_seed(seed)
                embedding, losses, snapshots = pretrain(config, graph, variant, args.protocol, args.probe_epochs)
                splits = get_splits(cpu_graph, dataset, args.protocol, args.roman_split, seed, args.smoke)
                splits = [(split, tuple(mask.to(device) for mask in masks)) for split, masks in splits]
                results = []
                set_random_seed(seed)
                for split, masks in splits:
                    result = probe(embedding, labels, masks, config)
                    results.append(dict(result, split=split))
                record = {
                    "seed": seed, "environment": environment, "last_pretrain_loss": losses,
                    "splits": results, "test_accuracy": statistics.mean(item["test_accuracy"] for item in results),
                }
                if args.probe_epochs:
                    diagnostics = []
                    for epoch in args.probe_epochs:
                        if epoch == config.max_epoch:
                            epoch_results, epoch_losses = results, losses
                        else:
                            snapshot, epoch_losses = snapshots[epoch]
                            snapshot = snapshot.to(device)
                            set_random_seed(seed)
                            epoch_results = [dict(probe(snapshot, labels, masks, config), split=split)
                                             for split, masks in splits]
                        diagnostics.append({"epoch": epoch, "pretrain_loss": epoch_losses, "splits": epoch_results})
                        print(json.dumps({"dataset": dataset, "variant": variant, "seed": seed,
                                          **diagnostics[-1]}), flush=True)
                    record["epoch_probes"] = diagnostics
                save_json(record_path, record)
                summarize(directory, dataset, variant, args.smoke, diagnostic=bool(args.probe_epochs))
            if not (directory / "summary.json").is_file():
                summarize(directory, dataset, variant, args.smoke, diagnostic=bool(args.probe_epochs))


if __name__ == "__main__":
    main()
