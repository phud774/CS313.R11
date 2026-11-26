"""DGMAE — cài đặt lại theo paper (Zheng et al., KDD 2025, arXiv:2506.19343).

Ánh xạ tới các phương trình trong paper:
  Eq. 1      : che node  -> mask_features()
  Eq. 2      : H^ = f_E(X^, A),  Z^ = f_D(H^, A)
  Eq. 3      : L_f = SCE(Z^[V^], X[V^], gamma1)
  Eq. 6      : w_ij = attention của encoder trên đồ thị bị che
  Eq. 7-9    : p_ij = min((1 - w_ij) * p_c, p_tau);  m_ij ~ Bernoulli(p_ij);
               x_i^D = sum_j m_ij (x_i - x_j) / sqrt(d_i d_j)   -> discrepancy_target()
  Eq. 10     : H = f_E(X, A),  Z = projector(H)
  Eq. 11     : z_i^D = z_i - z^_i
  Eq. 12     : L_d = SCE(Z^D[V \\ V^], X^D[V \\ V^], gamma2)
  Eq. 13     : L = (1 - lambda) L_f + lambda L_d
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat import GAT


def sce_loss(x, y, gamma):
    """Scaled Cosine Error: mean_i (1 - cos(x_i, y_i))^gamma (Eq. 3 / Eq. 12)."""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return (1 - (x * y).sum(-1)).pow(gamma).mean()


class DGMAE(nn.Module):
    SELECTIONS = ("inv_attn", "attn", "random", "degree", "all")

    def __init__(self, in_dim, num_hidden=512, num_layers=2, heads=4, heads_out=1,
                 in_drop=0.2, attn_drop=0.1, residual=False, norm=None,
                 mask_rate=0.5, replace_rate=0.0,
                 lam=0.1, gamma1=3.0, gamma2=3.0, p_c=0.3, p_tau=0.7,
                 selection="inv_attn", hop=1):
        super().__init__()
        assert selection in self.SELECTIONS, selection
        self.mask_rate, self.replace_rate = mask_rate, replace_rate
        self.lam, self.gamma1, self.gamma2 = lam, gamma1, gamma2
        self.p_c, self.p_tau, self.selection, self.hop = p_c, p_tau, selection, hop

        # f_E: encoder GAT, đầu ra num_hidden = heads * (num_hidden // heads)
        self.encoder = GAT(in_dim, num_hidden // heads, num_hidden // heads, num_layers, heads, heads,
                           in_drop, attn_drop, residual=residual, norm=norm, encoding=True)
        # f_D: decoder GAT 1 lớp, tái tạo về không gian đặc trưng gốc
        self.encoder_to_decoder = nn.Linear(num_hidden, num_hidden, bias=False)
        self.decoder = GAT(num_hidden, in_dim, in_dim, 1, heads, heads_out,
                           in_drop, attn_drop, residual=residual, norm=norm, encoding=False)
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        # project_theta: MLP 2 lớp đưa H về cùng không gian với Z^ (Eq. 10)
        self.projector = nn.Sequential(
            nn.Linear(num_hidden, num_hidden, bias=False), nn.PReLU(), nn.Dropout(0.2),
            nn.Linear(num_hidden, in_dim, bias=False))

    # ---------------------------------------------------------------- Eq. 1
    def mask_features(self, x):
        n = x.size(0)
        perm = torch.randperm(n, device=x.device)
        num_mask = int(self.mask_rate * n)
        mask_nodes = perm[:num_mask]
        x_masked = x.clone()
        if self.replace_rate > 0:  # chiến lược của GraphMAE: 1 phần node bị che được thay bằng node ngẫu nhiên
            num_noise = int(self.replace_rate * num_mask)
            p = torch.randperm(num_mask, device=x.device)
            token_nodes = mask_nodes[p[: num_mask - num_noise]]
            noise_nodes = mask_nodes[p[num_mask - num_noise:]]
            x_masked[noise_nodes] = x[torch.randperm(n, device=x.device)[:num_noise]]
        else:
            token_nodes = mask_nodes
        x_masked[token_nodes] = 0.0
        x_masked[token_nodes] += self.mask_token
        keep = torch.ones(n, dtype=torch.bool, device=x.device)
        keep[mask_nodes] = False
        return x_masked, mask_nodes, keep

    # ------------------------------------------------------------- Eq. 7-9
    @torch.no_grad()
    def discrepancy_target(self, x, graph, attn):
        """X^D: discrepancy gốc của đặc trưng, chỉ trên các cạnh được chọn."""
        ei, self_loop = graph["edge_index"], graph["self_loop"]
        src, dst = ei[:, ~self_loop]                       # j -> i, bỏ self-loop
        e = src.numel()
        n = x.size(0)
        if self.selection == "all":                       # không chọn cạnh: X^D = L_sym X = X - D^-1/2 A D^-1/2 X (Eq. 5)
            # Lưu ý: Eq. 9 với m_ij = 1 KHÔNG trùng Eq. 5 trừ khi đồ thị đều bậc -> dùng đúng Eq. 5 ở đây.
            dis = graph["deg_inv_sqrt"]
            A_norm = torch.sparse_coo_tensor(torch.stack([dst, src]), dis[src] * dis[dst], (n, n), is_coalesced=True)
            h = x
            for _ in range(self.hop):
                h = h - torch.sparse.mm(A_norm, h)
            return h
        else:
            if self.selection == "inv_attn":              # paper: p = min((1 - w) p_c, p_tau)
                w = attn[~self_loop].mean(1)
                p = (1 - w) * self.p_c
            elif self.selection == "attn":                # biến thể Table 6: dùng w thay vì 1 - w
                w = attn[~self_loop].mean(1)
                p = w * self.p_c
            elif self.selection == "random":              # biến thể Table 6: xác suất đều
                p = torch.full((e,), self.p_c, device=x.device)
            else:                                         # "degree": ưu tiên cạnh nối node bậc cao
                c = graph["deg"][src] + graph["deg"][dst]
                p = c / c.mean() * self.p_c
            p = p.clamp(max=self.p_tau)
            m = torch.bernoulli(p)                        # Eq. 8
        # x_i^D = sum_j c_ij (x_i - x_j) = (sum_j c_ij) x_i - (C x)_i,  c_ij = m_ij / sqrt(d_i d_j)
        coef = m * graph["deg_inv_sqrt"][src] * graph["deg_inv_sqrt"][dst]
        C =torch.sparse_coo_tensor(torch.stack([dst, src]), coef, (n, n), is_coalesced=True)
        row_sum = torch.zeros(n, device=x.device).index_add_(0, dst, coef).unsqueeze(-1)
        h = x
        for _ in range(self.hop):                         # hop=1 đúng Eq. 9; hop=2 áp toán tử 2 lần
            h = row_sum * h - torch.sparse.mm(C, h)
        return h

    # -------------------------------------------------------------- forward
    def forward(self, x, graph):
        ei = graph["edge_index"]
        x_masked, mask_nodes, keep = self.mask_features(x)

        # Nhánh 1 — tái tạo đặc trưng (Eq. 2-3)
        h_hat, attn = self.encoder(x_masked, ei)
        rep = self.encoder_to_decoder(h_hat)
        rep[mask_nodes] = 0.0                              # re-mask trước decoder (GraphMAE)
        z_hat, _ = self.decoder(rep, ei)
        loss_f = sce_loss(z_hat[mask_nodes], x[mask_nodes], self.gamma1)

        if self.lam <= 0:                                  # lambda = 0  <=>  GraphMAE
            return loss_f, {"loss_f": loss_f.item(), "loss_d": 0.0}

        # Nhánh 2 — tái tạo discrepancy (Eq. 10-12)
        h, _ = self.encoder(x, ei)                         # đồ thị không che, encoder dùng chung
        z = self.projector(h)
        z_d = z - z_hat                                    # contextual discrepancy (Eq. 11)
        x_d = self.discrepancy_target(x, graph, attn.detach())
        loss_d = sce_loss(z_d[keep], x_d[keep], self.gamma2)

        loss = (1 - self.lam) * loss_f + self.lam * loss_d   # Eq. 13
        return loss, {"loss_f": loss_f.item(), "loss_d": loss_d.item()}

    @torch.no_grad()
    def embed(self, x, graph):
        return self.encoder(x, graph["edge_index"])[0]
