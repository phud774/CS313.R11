"""Đánh giá biểu diễn đã đóng băng.

- Phân loại node (Mục 5.3): linear probing — logistic regression trên embedding cố định,
  chọn epoch theo accuracy tập VAL, báo cáo accuracy tập TEST tại epoch đó.
- Phân cụm node: K-means (10 lần khởi tạo khác nhau), độ đo ACC, NMI, ARI, F1 (macro).
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


def linear_probe(emb, y, train, val, test, num_classes, lr=0.01, weight_decay=1e-4, epochs=300,
                 return_oracle=False):
    """Trả về list accuracy test (%) cho từng split (cột của mask), chọn epoch theo tập val.

    return_oracle=True: trả thêm accuracy khi chọn epoch theo TẬP TEST (như code tác giả,
    evaluation.py: best_val_acc = test_acc) — chỉ dùng để đo mức lạc quan, không phải kết quả chính.
    """
    emb = emb.detach()
    accs, oracle = [], []
    for s in range(train.size(1)):
        tr, va, te = train[:, s], val[:, s], test[:, s]
        clf = nn.Linear(emb.size(1), num_classes).to(emb.device)
        opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
        best_val, best_test, best_test_any = -1.0, 0.0, 0.0
        for _ in range(epochs):
            clf.train()
            loss = nn.functional.cross_entropy(clf(emb[tr]), y[tr])
            opt.zero_grad()
            loss.backward()
            opt.step()
            clf.eval()
            with torch.no_grad():
                pred = clf(emb).argmax(1)
                va_acc = (pred[va] == y[va]).float().mean().item()
                te_acc = (pred[te] == y[te]).float().mean().item()
            if va_acc >= best_val:
                best_val, best_test = va_acc, te_acc
            best_test_any = max(best_test_any, te_acc)
        accs.append(best_test * 100)
        oracle.append(best_test_any * 100)
    return (accs, oracle) if return_oracle else accs


def cluster_acc_f1(y_true, y_pred):
    """ACC và F1-macro sau khi ghép cụm <-> lớp bằng thuật toán Hungarian."""
    k = max(y_true.max(), y_pred.max()) + 1
    cost = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cost[p, t] += 1
    rows, cols = linear_sum_assignment(-cost)
    mapping = dict(zip(rows, cols))
    mapped = np.array([mapping[p] for p in y_pred])
    return (mapped == y_true).mean(), f1_score(y_true, mapped, average="macro")


def clustering(emb, y, num_classes, runs=10):
    x = emb.detach().cpu().numpy()
    yt = y.cpu().numpy()
    res = {"acc": [], "nmi": [], "ari": [], "f1": []}
    for r in range(runs):
        pred = KMeans(n_clusters=num_classes, n_init=10, random_state=r).fit_predict(x)
        acc, f1 = cluster_acc_f1(yt, pred)
        res["acc"].append(acc * 100)
        res["f1"].append(f1 * 100)
        res["nmi"].append(normalized_mutual_info_score(yt, pred) * 100)
        res["ari"].append(adjusted_rand_score(yt, pred) * 100)
    return res
