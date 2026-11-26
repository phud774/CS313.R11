"""Pre-train DGMAE rồi đánh giá: phân loại node (linear probe) và/hoặc phân cụm (K-means).

Ví dụ:
  python train_node.py --dataset texas --seeds 0 1 2 --task both
  python train_node.py --dataset texas --lam 0            # GraphMAE (bỏ L_d)
  python train_node.py --dataset texas --selection all     # bỏ chọn cạnh thích ứng (X^D = L_sym X)
  python train_node.py --dataset texas --mask_rate 0.3 --tag mask0.3
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import yaml

from dgmae.data import load_node_dataset
from dgmae.evaluate import clustering, linear_probe
from dgmae.model import DGMAE
from dgmae.utils import set_seed

OVERRIDES = ["lam", "p_c", "p_tau", "gamma1", "gamma2", "weight_decay", "lr", "mask_rate", "num_layers",
             "num_hidden", "heads", "max_epoch", "replace_rate", "selection", "hop"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--task", choices=["cls", "cluster", "both"], default="cls")
    p.add_argument("--config", default="configs/node.yml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="dgmae")
    p.add_argument("--out", default="results")
    p.add_argument("--save_emb", action="store_true")
    p.add_argument("--log_every", type=int, default=100)
    for k in OVERRIDES:
        p.add_argument(f"--{k}", type=str if k == "selection" else (int if k in
                       ("num_layers", "num_hidden", "heads", "max_epoch", "hop") else float), default=None)
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)[args.dataset]
    cfg.setdefault("selection", "inv_attn")
    cfg.setdefault("hop", 1)
    for k in OVERRIDES:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    return args, cfg


def build_model(in_dim, cfg):
    return DGMAE(in_dim, num_hidden=cfg["num_hidden"], num_layers=cfg["num_layers"], heads=cfg["heads"],
                 residual=cfg.get("residual", False), norm=cfg.get("norm"),
                 mask_rate=cfg["mask_rate"], replace_rate=cfg.get("replace_rate", 0.0),
                 lam=cfg["lam"], gamma1=cfg["gamma1"], gamma2=cfg["gamma2"],
                 p_c=cfg["p_c"], p_tau=cfg["p_tau"], selection=cfg["selection"], hop=cfg["hop"])


def pretrain(model, x, graph, cfg, log_every):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    T = cfg["max_epoch"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (1 + math.cos(e * math.pi / T)) * 0.5)
    history = []
    for epoch in range(T):
        model.train()
        loss, parts = model(x, graph)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        history.append({"epoch": epoch, "loss": loss.item(), **parts})
        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch + 1:5d} | loss {loss.item():.4f} | L_f {parts['loss_f']:.4f} | L_d {parts['loss_d']:.4f}")
    return history


def main():
    args, cfg = get_args()
    x, y, graph, (train, val, test), info = load_node_dataset(args.dataset, device=args.device)
    print(f"[{args.tag}] {info}")
    print(f"config: {cfg}")

    cls_all, cls_oracle, clu_all, per_seed = [], [], {"acc": [], "nmi": [], "ari": [], "f1": []}, []
    for seed in args.seeds:
        set_seed(seed)
        model = build_model(x.size(1), cfg).to(args.device)
        t0 = time.time()
        history = pretrain(model, x, graph, cfg, args.log_every)
        train_time = time.time() - t0
        model.eval()
        emb = model.embed(x, graph)
        rec = {"seed": seed, "train_time_s": train_time,
               "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None}
        if args.task in ("cls", "both"):
            accs, oracle = linear_probe(emb, y, train, val, test, info["num_classes"], return_oracle=True)
            cls_all += accs
            cls_oracle += oracle
            rec["cls_acc_per_split"] = accs
            rec["cls_acc_test_selected_per_split"] = oracle  # chọn theo test, chỉ để đối chiếu
            print(f"seed {seed}: node cls acc = {np.mean(accs):.2f} ± {np.std(accs):.2f}  ({train_time:.0f}s)")
        if args.task in ("cluster", "both"):
            res = clustering(emb, y, info["num_classes"])
            for k in clu_all:
                clu_all[k] += res[k]
            rec["cluster"] = res
            print("seed {}: clustering ".format(seed) + ", ".join(f"{k}={np.mean(v):.2f}" for k, v in res.items()))
        if args.save_emb:
            os.makedirs(f"{args.out}/emb", exist_ok=True)
            torch.save({"emb": emb.cpu(), "y": y.cpu()}, f"{args.out}/emb/{args.dataset}_{args.tag}_s{seed}.pt")
        rec["history"] = history[:: max(1, len(history) // 200)]
        per_seed.append(rec)

    summary = {"dataset": args.dataset, "tag": args.tag, "config": cfg, "info": info, "seeds": args.seeds}
    if cls_all:
        summary["cls_acc_mean"], summary["cls_acc_std"] = float(np.mean(cls_all)), float(np.std(cls_all))
        summary["cls_acc_test_selected_mean"] = float(np.mean(cls_oracle))
        summary["cls_acc_test_selected_std"] = float(np.std(cls_oracle))
        print(f"==> {args.dataset} [{args.tag}] node classification: {np.mean(cls_all):.2f} ± {np.std(cls_all):.2f}")
        print(f"    (đối chiếu) chọn theo test: {np.mean(cls_oracle):.2f} ± {np.std(cls_oracle):.2f}")
    if clu_all["acc"]:
        summary["cluster"] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in clu_all.items()}
        print(f"==> {args.dataset} [{args.tag}] clustering: " +
              ", ".join(f"{k}={m:.2f}±{s:.2f}" for k, (m, s) in summary["cluster"].items()))
    summary["runs"] = per_seed
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.dataset}_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
