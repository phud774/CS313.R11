"""GAT tự cài đặt bằng PyTorch thuần (không dùng DGL/PyG layer).

Mỗi lớp trả về cả trọng số attention w_ij (Eq. 6 trong paper) để module
chọn discrepancy thích ứng có thể dùng lại.
Quy ước: edge_index[0] = nguồn j, edge_index[1] = đích i (thông điệp j -> i),
attention được chuẩn hóa softmax trên các cạnh đi vào node đích i.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import make_norm


def edge_softmax(e: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax của điểm cạnh e [E, H] theo từng node đích."""
    idx = dst.unsqueeze(-1).expand_as(e)
    e_max = torch.full((num_nodes, e.size(1)), float("-inf"), device=e.device, dtype=e.dtype)
    e_max = e_max.scatter_reduce(0, idx, e, reduce="amax", include_self=True)
    e = torch.exp(e - e_max[dst])
    denom = torch.zeros((num_nodes, e.size(1)), device=e.device, dtype=e.dtype).index_add_(0, dst, e)
    return e / (denom[dst] + 1e-16)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads, feat_drop=0.0, attn_drop=0.0,
                 negative_slope=0.2, residual=False, activation=None, norm=None, concat=True):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.fc = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn_l = nn.Parameter(torch.empty(1, heads, out_dim))
        self.attn_r = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Identity() if in_dim == heads * out_dim else nn.Linear(in_dim, heads * out_dim, bias=False)
        else:
            self.res_fc = None
        out_total = heads * out_dim if concat else out_dim
        self.norm = make_norm(norm, out_total)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, x, edge_index):
        n = x.size(0)
        src, dst = edge_index
        h = self.feat_drop(x)
        f = self.fc(h).view(n, self.heads, self.out_dim)              # W h
        el = (f * self.attn_l).sum(-1)                                 # a_l^T W h_j
        er = (f * self.attn_r).sum(-1)                                 # a_r^T W h_i
        e = self.leaky_relu(el[src] + er[dst])                         # [E, H]
        alpha = edge_softmax(e, dst, n)                                # w_ij (Eq. 6)
        a = self.attn_drop(alpha)
        idx = torch.stack([dst, src])                                  # ma trận thưa [i, j] = w_ij
        # .contiguous() BẮT BUỘC: trên CUDA, sparse.mm với lát cắt không liên tục (a[:, k], f[:, k]) cho kết quả sai.
        out = torch.stack([torch.sparse.mm(torch.sparse_coo_tensor(idx, a[:, k].contiguous(), (n, n), is_coalesced=True),
                                           f[:, k].contiguous()) for k in range(self.heads)], 1)   # sum_j w_ij W h_j
        out = out + self.bias.view(1, self.heads, self.out_dim)
        if self.res_fc is not None:
            out = out + self.res_fc(h).view(n, -1, self.out_dim)
        out = out.flatten(1) if self.concat else out.mean(1)
        out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out, alpha


class GAT(nn.Module):
    """GAT nhiều lớp. encoding=True: lớp cuối có activation/residual/norm (giống GraphMAE)."""

    def __init__(self, in_dim, hidden_per_head, out_per_head, num_layers, heads, heads_out,
                 feat_drop, attn_drop, negative_slope=0.2, residual=False, norm=None, encoding=True):
        super().__init__()
        self.layers = nn.ModuleList()
        last_act = nn.PReLU() if encoding else None
        last_res = residual and encoding
        last_norm = norm if encoding else None
        if num_layers == 1:
            self.layers.append(GATLayer(in_dim, out_per_head, heads_out, feat_drop, attn_drop,
                                        negative_slope, last_res, last_act, last_norm))
        else:
            self.layers.append(GATLayer(in_dim, hidden_per_head, heads, feat_drop, attn_drop,
                                        negative_slope, residual, nn.PReLU(), norm))
            for _ in range(1, num_layers - 1):
                self.layers.append(GATLayer(hidden_per_head * heads, hidden_per_head, heads, feat_drop,
                                            attn_drop, negative_slope, residual, nn.PReLU(), norm))
            self.layers.append(GATLayer(hidden_per_head * heads, out_per_head, heads_out, feat_drop,
                                        attn_drop, negative_slope, last_res, last_act, last_norm))

    def forward(self, x, edge_index):
        attn = None
        for layer in self.layers:
            x, attn = layer(x, edge_index)
        return x, attn  # attention của lớp cuối, [E, heads]
