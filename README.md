# Khung slide — DGMAE: Discrepancy-Aware Graph Mask Auto-Encoder

> Đồ án môn học · Thuyết trình phương pháp + Implement lại paper
> Paper: Ziyu Zheng, Yaming Yang, Ziyu Guan, Wei Zhao, Weigang Lu — *KDD 2025* — arXiv:2506.19343
> Code tác giả: https://github.com/zhengziyu77/DGMAE

---

## Thông tin chung

**Thời lượng:** ~25 phút thuyết trình + 5–10 phút hỏi đáp · **27 slide** · **3 người thuyết trình**

| Phần | Slide | Thuyết trình | Làm nội dung | Thời gian |
|---|---|---|---|---|
| 1. Mở đầu + Nền tảng + Kiến trúc | 1–10 | **A (TV2)** | TV2, TV3 | ~8' |
| 2. Phương pháp chính + Implement | 11–17 | **B (TV4)** | TV4, TV1 | ~7.5' |
| 3. Thực nghiệm + Phân tích + Kết luận | 18–27 | **C (TV5)** | TV5, TV6, TV7 | ~9.5' |

**Quy ước trong file này**
- `[ĐIỀN]` = chỗ cần điền số liệu / kết quả của nhóm.
- `[KIỂM TRA]` = chi tiết cần đối chiếu lại với paper trước khi đưa lên slide.
- **Trên slide** = nội dung hiển thị · **Lời thoại** = ý người nói (không đưa lên slide) · **Cần chuẩn bị** = hình/bảng.

**Bảng ký hiệu chung (dùng thống nhất ở mọi slide)**

| Ký hiệu | Ý nghĩa |
|---|---|
| G = (V, E), A | Đồ thị, ma trận kề |
| X, x_i | Ma trận đặc trưng node, đặc trưng node i |
| d_i, N(i) | Bậc và tập láng giềng của node i |
| V̂, X̂ | Tập node bị che, đặc trưng sau khi che |
| f_E, f_D | Encoder, decoder (GNN) |
| Ĥ, Ẑ | Biểu diễn của đồ thị bị che (sau encoder / sau decoder) |
| H, Z | Biểu diễn của đồ thị không che (sau encoder / sau projector) |
| x_i^D, z_i^D | Discrepancy gốc (mục tiêu) / discrepancy theo ngữ cảnh |
| w_ij, p_ij, m_ij | Trọng số attention, xác suất chọn cạnh, mask chọn cạnh |
| L_f, L_d | Loss tái tạo đặc trưng / loss tái tạo discrepancy |
| γ₁, γ₂, λ, p_c, p_τ | Siêu tham số |

---

# PHẦN 1 — MỞ ĐẦU, NỀN TẢNG, KIẾN TRÚC
**Thuyết trình: A (TV2) · Làm nội dung: TV2, TV3 · ~8 phút**

---

### Slide 1 — Tiêu đề

**Trên slide**
- **Discrepancy-Aware Graph Mask Auto-Encoder (DGMAE)**
- Zheng et al., KDD 2025
- Tên môn học · Tên nhóm · Giảng viên hướng dẫn · Ngày `[ĐIỀN]`

**Lời thoại:** Chào, giới thiệu nhóm và tên paper trong 1 câu.

---

### Slide 2 — Thành viên & phân công

**Trên slide**

| Thành viên | Vai trò |
|---|---|
| TV1 `[ĐIỀN tên]` | Trưởng nhóm · Repo, pipeline, tích hợp code |
| TV2 `[ĐIỀN tên]` | Kiến thức nền · Baseline GraphMAE · **Thuyết trình phần 1** |
| TV3 `[ĐIỀN tên]` | Masking, encoder/decoder, loss L_f |
| TV4 `[ĐIỀN tên]` | Module discrepancy, chọn cạnh thích ứng, L_d · **Thuyết trình phần 2** |
| TV5 `[ĐIỀN tên]` | Thực nghiệm phân loại node · **Thuyết trình phần 3** |
| TV6 `DATKINGKHUNG` | Thực nghiệm phân cụm + phân loại đồ thị |
| TV7 `[ĐIỀN tên]` | Ablation, độ nhạy, trực quan hóa, template slide |

**Lời thoại:** Lướt nhanh, nói rõ ai trả lời câu hỏi phần nào ở cuối.

---

### Slide 3 — Mục lục

**Trên slide**
1. Bối cảnh & vấn đề
2. Kiến thức nền: GNN, heterophily, Graph Masked Auto-Encoder
3. Phương pháp DGMAE
4. Implement của nhóm
5. Thực nghiệm & phân tích
6. Kết luận

---

### Slide 4 — Bối cảnh & động lực

**Trên slide**
- Dữ liệu đồ thị ở khắp nơi: mạng xã hội, trích dẫn, phân tử, web…
- Nhãn đắt đỏ → **học tự giám sát (self-supervised)** trên đồ thị
- Hai hướng chính: **contrastive** (DGI, GRACE, BGRL…) và **generative** (GraphMAE, MaskGAE…)
- Graph Masked Auto-Encoder (GMAE) đang cho kết quả rất tốt… **nhưng chủ yếu trên đồ thị homophilic**
- ❓ Câu hỏi của paper: *Làm sao để GMAE hoạt động tốt cả trên đồ thị heterophilic?*

**Lời thoại:** Đặt vấn đề: GMAE mạnh, nhưng có một điểm yếu cụ thể mà paper này nhắm vào.

---

### Slide 5 — GNN & cơ chế truyền thông điệp

**Trên slide**
- Mỗi lớp GNN: node **tổng hợp thông tin từ láng giềng** rồi cập nhật biểu diễn
- GCN: trung bình có chuẩn hóa theo bậc `x_j / (√d_i √d_j)`
- GAT: trọng số **attention** `w_ij` học được cho từng cạnh
- → Bản chất là **làm mượt (smoothing)**: node càng giống láng giềng

**Cần chuẩn bị:** Hình minh họa 1 node nhận thông điệp từ láng giềng (vẽ lại).

**Lời thoại:** Nhấn mạnh "làm mượt" vì đây là gốc của vấn đề ở slide sau. Nhắc rằng attention w_ij sẽ được DGMAE tái sử dụng.

---

### Slide 6 — Homophily vs. Heterophily

**Trên slide**
- **Homophily:** node nối nhau thường cùng lớp (Cora, Citeseer, Pubmed)
- **Heterophily:** node nối nhau thường *khác* lớp (Texas, Cornell, Wisconsin, Chameleon, Squirrel, Actor…)
- Ví dụ trong paper: học giả A (lĩnh vực X) cộng tác với học giả B (lĩnh vực Y)
  - Che A rồi tái tạo từ láng giềng → đặc trưng của B **làm nhiễu** kết quả
- Edge homophily ratio: `h = |{(i,j) ∈ E : y_i = y_j}| / |E|`

**Cần chuẩn bị:** Hình 2 đồ thị nhỏ cạnh nhau (homophilic / heterophilic), tô màu theo lớp. Bảng nhỏ homophily ratio của vài dataset `[ĐIỀN]`.

---

### Slide 7 — Graph Masked Auto-Encoder (GraphMAE)

**Trên slide**
- Pipeline: **Che node → Encoder → Decoder → Tái tạo đặc trưng của node bị che**
- Loss: Scaled Cosine Error (SCE) giữa đặc trưng gốc và đặc trưng tái tạo
- Sau pre-train: bỏ decoder, dùng biểu diễn của encoder cho tác vụ downstream
- Điểm mạnh: nắm tốt thông tin láng giềng, cấu trúc cục bộ

**Cần chuẩn bị:** Sơ đồ pipeline GraphMAE (vẽ lại, dùng chung style với sơ đồ DGMAE ở slide 9).

---

### Slide 8 — Vấn đề: GMAE xóa nhòa sự khác biệt

**Trên slide**
- GMAE chỉ học cách **khôi phục từ ngữ cảnh** → bỏ qua **sự khác biệt (discrepancy)** giữa các node
- Trên đồ thị heterophilic: các cặp node nối nhau nhưng khác lớp bị kéo **gần nhau** trong không gian biểu diễn
- → Biểu diễn khó phân biệt → kết quả downstream kém
- Bằng chứng (Figure 2 của paper): độ tương đồng của các cặp node heterophilic **tăng lên** sau khi học bằng GMAE

**Cần chuẩn bị:** Tái hiện Figure 2 bằng kết quả của nhóm (GraphMAE baseline của TV2) nếu kịp; nếu không, dùng hình paper + ghi nguồn.

**Lời thoại:** Câu chuyển: *"Vậy làm sao vừa tận dụng ngữ cảnh, vừa giữ được sự khác biệt? Đó là ý tưởng của DGMAE."*

---

### Slide 9 — Tổng quan kiến trúc DGMAE

**Trên slide**
- **Hai nhánh, dùng chung một encoder:**
  1. **Nhánh tái tạo đặc trưng** (giống GraphMAE) → loss **L_f**
  2. **Nhánh tái tạo discrepancy** (mới) → loss **L_d**
- Đầu vào: đồ thị bị che (nhánh 1) + đồ thị gốc không che (nhánh 2)
- Thành phần mới: mục tiêu discrepancy, chọn cạnh thích ứng, projector

**Cần chuẩn bị:** **Sơ đồ kiến trúc vẽ lại** (TV3 + TV4), tô màu khác cho phần mới. Tham khảo `figure3-model.png` trong repo tác giả.

**Lời thoại:** Giới thiệu khung tổng thể; phần mới (tô màu) sẽ được B trình bày chi tiết.

---

### Slide 10 — Nhánh 1: Masking, Encoder/Decoder, L_f

**Trên slide**
- **Masking (Eq. 1):** chọn tập node V̂ theo phân phối Bernoulli (tỉ lệ che = mask ratio), đặt đặc trưng của chúng về 0 → X̂
- **Encoder / Decoder (Eq. 2):**
  `Ĥ = f_E(X̂, A)`  ·  `Ẑ = f_D(Ĥ, A)`  (decoder là GAT)
- **Loss tái tạo đặc trưng (Eq. 3) — Scaled Cosine Error:**

  `L_f = (1/|V̂|) · Σ_{i∈V̂} (1 − cos(x_i, ẑ_i))^γ₁`, với γ₁ > 1

- γ₁ > 1 → giảm trọng số mẫu dễ, tập trung vào mẫu khó

**Lời thoại:** Giải thích ngắn vì sao dùng cosine thay MSE (ổn định với độ lớn vector, γ giúp tập trung mẫu khó). Chuyển lời cho B.

---

# PHẦN 2 — PHƯƠNG PHÁP CHÍNH & IMPLEMENT
**Thuyết trình: B (TV4) · Làm nội dung: TV4, TV1 · ~7.5 phút**

---

### Slide 11 — Ý tưởng chính

**Trên slide**
- GMAE truyền thống: **"khôi phục sự nhất quán"** với láng giềng
- DGMAE: **"bảo toàn sự khác biệt"** giữa node và láng giềng
- Ý tưởng: bắt mô hình **tái tạo luôn cả độ chênh lệch** giữa node và láng giềng, không chỉ đặc trưng của node
- → Biểu diễn trong không gian ít chiều vẫn giữ được discrepancy → dễ phân biệt hơn

**Lời thoại:** Slide quan trọng nhất về mặt ý tưởng — nói chậm, dùng lại ví dụ học giả A/B.

---

### Slide 12 — Mục tiêu discrepancy gốc

**Trên slide**
- Discrepancy của node i so với láng giềng (Eq. 4):

  `x_i^D = x_i − Σ_{j∈N(i)} x_j / (√d_i · √d_j)`

- Dạng ma trận (Eq. 5): `X^D = L_sym · X`, với `L_sym = I − D^{-1/2} A D^{-1/2}`
- Diễn giải: Laplacian chuẩn hóa = **bộ lọc thông cao** → giữ thành phần tần số cao (sự khác biệt), ngược với GNN thông thường là **lọc thông thấp** (làm mượt)
- Lưu ý: x^D là **mục tiêu tái tạo**, không phải đầu vào encoder

**Lời thoại:** Liên hệ lại slide 5: GNN làm mượt = thông thấp; DGMAE bổ sung tín hiệu thông cao.

---

### Slide 13 — Chọn discrepancy thích ứng (Adaptive Discrepancy Selection)

**Trên slide**
- Không phải láng giềng nào cũng "khác biệt" đáng kể → chọn cạnh có chủ đích
- Lấy trọng số attention `w_ij` từ encoder (GAT) (Eq. 6)
- Xác suất chọn cạnh (Eq. 7):

  `p_ij = min( (1 − w_ij) · p_c , p_τ )`

  - p_c: điều chỉnh số lượng cạnh được chọn · p_τ: ngưỡng chặn trên
- Lấy mẫu (Eq. 8): `m_ij ~ Bernoulli(p_ij)`
- Discrepancy sau khi chọn (Eq. 9):

  `x_i^D = Σ_{j∈N(i)} m_ij · (x_i − x_j) / (√d_i · √d_j)`

- **Trực giác:** attention thấp ⇔ cặp node ít giống nhau ⇔ **ưu tiên chọn**

**Cần chuẩn bị:** Hình minh họa 1 node với các cạnh có độ dày theo w_ij, cạnh được chọn tô màu.

---

### Slide 14 — Projector, Contextual Discrepancy & L_d

**Trên slide**
- Mã hóa đồ thị **không che**: `H = f_E(X, A)`
- **Projector** (MLP 2 lớp) đưa H về cùng không gian với Ẑ (Eq. 10): `Z = project_θ(H)`
- **Contextual discrepancy** (Eq. 11): `z_i^D = z_i − ẑ_i`
  - = chênh lệch giữa biểu diễn "thấy đầy đủ" và biểu diễn "tái tạo từ ngữ cảnh"
- **Loss discrepancy** (Eq. 12) — cùng dạng SCE:

  `L_d = mean_i (1 − cos(z_i^D, x_i^D))^γ₂`, với γ₂ > 1

  `[KIỂM TRA]` tập node lấy trung bình trong Eq. 12

**Lời thoại:** Giải thích vì sao cần projector: H và Ẑ nằm ở hai không gian khác nhau (sau encoder vs sau decoder).

---

### Slide 15 — Loss tổng & thuật toán huấn luyện

**Trên slide**
- Loss tổng: `L = L_f + λ · L_d`  `[KIỂM TRA]` công thức chính xác ở Mục 4.4
- **Một bước huấn luyện:**
  ```
  1. Lấy mẫu V̂ ~ Bernoulli(mask_ratio); X̂ ← X, X̂[V̂] ← 0
  2. Ĥ ← f_E(X̂, A);  Ẑ ← f_D(Ĥ, A)
  3. L_f ← SCE(X[V̂], Ẑ[V̂], γ₁)
  4. H ← f_E(X, A);  Z ← project_θ(H)
  5. w ← attention của encoder;  p ← min((1−w)·p_c, p_τ);  m ~ Bernoulli(p)
  6. X^D ← Σ_j m_ij (x_i − x_j)/(√d_i √d_j)
  7. Z^D ← Z − Ẑ;  L_d ← SCE(X^D, Z^D, γ₂)
  8. L ← L_f + λ·L_d;  cập nhật tham số
  ```
- Suy luận: chỉ dùng encoder → biểu diễn H cho tác vụ downstream

---

### Slide 16 — Implement của nhóm: cấu trúc code

**Trên slide**
- Framework: `[ĐIỀN: PyTorch + PyG / DGL]` · Phần cứng: `[ĐIỀN]`
- Cấu trúc repo:
  ```
  dgmae/
  ├── data/          # loader cho các dataset (TV1)
  ├── models/
  │   ├── encoder.py       # GAT encoder (TV3)
  │   ├── decoder.py       # GAT decoder (TV3)
  │   ├── discrepancy.py   # L_sym·X, chọn cạnh thích ứng (TV4)
  │   └── dgmae.py         # ghép 2 nhánh, projector, loss (TV4)
  ├── eval/          # linear probe, k-means, graph cls (TV5, TV6)
  ├── analysis/      # ablation, sensitivity, t-SNE (TV7)
  └── configs/       # siêu tham số theo dataset
  ```
- Code tự viết; code tác giả (`main_transductive.py`, `main_cl.py`, `main_graph.py`) chỉ dùng để **đối chiếu kết quả**

---

### Slide 17 — Implement: khó khăn & đối chiếu

**Trên slide**
- Khó khăn gặp phải & cách xử lý `[ĐIỀN]`, ví dụ:
  - Lấy attention của encoder để chọn cạnh (cần trả attention ra khỏi lớp GAT)
  - Tính L_sym·X hiệu quả với ma trận thưa
  - Siêu tham số khác nhau theo từng dataset
- Kiểm tra tính đúng: so sánh với code tác giả trên Cora và Texas `[ĐIỀN số]`
- Thời gian huấn luyện / dataset `[ĐIỀN]`

**Lời thoại:** Câu chuyển sang C: *"Vậy phương pháp này có thực sự hiệu quả không? Phần thực nghiệm sẽ trả lời."*

---

# PHẦN 3 — THỰC NGHIỆM, PHÂN TÍCH, KẾT LUẬN
**Thuyết trình: C (TV5) · Làm nội dung: TV5, TV6, TV7 · ~9.5 phút**

---

### Slide 18 — Thiết lập thực nghiệm

**Trên slide**
- **Paper:** 17 dataset, 3 tác vụ (phân loại node, phân cụm node, phân loại đồ thị)
- **Nhóm chọn tái hiện** (lý do: giới hạn tài nguyên tính toán):
  - Heterophilic: Texas, Cornell, Wisconsin, Chameleon, Squirrel, Actor
  - Homophilic: Cora, Citeseer, Pubmed
  - Phân loại đồ thị: PROTEINS `[ĐIỀN thêm nếu có]`
- **Protocol:** pre-train tự giám sát → đóng băng encoder → đánh giá trên biểu diễn `[KIỂM TRA: linear probing, cách chia tập ở Mục 5.3 / Phụ lục]`
- Số lần chạy: `[ĐIỀN]` seed, báo cáo trung bình ± độ lệch chuẩn
- **Baseline:**
  - Contrastive: DGI, GCA, CCA-SSG, BGRL, SP-GCL, GREET, GraphACL, DSSL
  - Generative: VGAE, GraphMAE, SeeGera, MaskGAE, NWR-GAE, AUG-MAE, Bandana
  - Nhóm tự chạy: GraphMAE `[ĐIỀN thêm]`

---

### Slide 19 — Phân loại node: đồ thị heterophilic

**Trên slide** (accuracy %, trung bình ± độ lệch chuẩn)

| Dataset | GraphMAE (nhóm chạy) | DGMAE (paper) | DGMAE (nhóm) | Chênh lệch |
|---|---|---|---|---|
| Texas | `[ĐIỀN]` | 88.11 ± 5.16 | `[ĐIỀN]` | `[ĐIỀN]` |
| Cornell | `[ĐIỀN]` | 78.65 ± 4.59 | `[ĐIỀN]` | `[ĐIỀN]` |
| Wisconsin | `[ĐIỀN]` | 88.43 ± 3.56 | `[ĐIỀN]` | `[ĐIỀN]` |
| Chameleon | `[ĐIỀN]` | 75.50 ± 1.17 | `[ĐIỀN]` | `[ĐIỀN]` |
| Squirrel | `[ĐIỀN]` | 72.47 ± 1.77 | `[ĐIỀN]` | `[ĐIỀN]` |
| Actor | `[ĐIỀN]` | 36.61 ± 0.74 | `[ĐIỀN]` | `[ĐIỀN]` |

*(Paper còn có Crocodile 75.48 ± 0.50 và Roman-empire 76.62 ± 0.41)*

**Lời thoại:** Nhấn mạnh mức cải thiện so với GraphMAE trên heterophilic — đây là luận điểm chính của paper.

---

### Slide 20 — Phân loại node: đồ thị homophilic

**Trên slide**

| Dataset | GraphMAE (nhóm chạy) | DGMAE (paper) | DGMAE (nhóm) |
|---|---|---|---|
| Cora | `[ĐIỀN]` | `[ĐIỀN từ Bảng 2]` | `[ĐIỀN]` |
| Citeseer | `[ĐIỀN]` | `[ĐIỀN từ Bảng 2]` | `[ĐIỀN]` |
| Pubmed | `[ĐIỀN]` | `[ĐIỀN từ Bảng 2]` | `[ĐIỀN]` |

- Thông điệp: DGMAE **không đánh đổi** hiệu năng trên homophilic để đạt được trên heterophilic `[KIỂM TRA với số thực tế]`

---

### Slide 21 — Phân cụm node

**Trên slide**
- Biểu diễn từ encoder → K-means → so với nhãn thật
- Độ đo: ACC, NMI, ARI, F1

| Dataset | ACC | NMI | ARI | F1 |
|---|---|---|---|---|
| `[ĐIỀN]` | `[ĐIỀN]` | `[ĐIỀN]` | `[ĐIỀN]` | `[ĐIỀN]` |

- So sánh với GraphMAE và số trong paper `[ĐIỀN]`

---

### Slide 22 — Phân loại đồ thị

**Trên slide**
- Pre-train trên tập đồ thị → pooling biểu diễn node thành biểu diễn đồ thị → bộ phân loại `[KIỂM TRA cách pooling & classifier]`

| Dataset | GraphMAE | DGMAE (paper) | DGMAE (nhóm) |
|---|---|---|---|
| PROTEINS | `[ĐIỀN]` | `[ĐIỀN]` | `[ĐIỀN]` |

---

### Slide 23 — Ablation study

**Trên slide**

| Biến thể | Texas | Chameleon | Cora | `[ĐIỀN]` |
|---|---|---|---|---|
| DGMAE đầy đủ | | | | |
| Bỏ L_d (≈ GraphMAE) | | | | |
| Bỏ chọn cạnh thích ứng (dùng toàn bộ L_sym·X) | | | | |
| Chọn cạnh ngẫu nhiên (không dùng attention) | | | | |
| Bỏ projector | | | | |

- Kết luận: thành phần nào đóng góp nhiều nhất? `[ĐIỀN]`

**Lời thoại:** Đây là slide chứng minh **từng** thành phần mới đều có ích.

---

### Slide 24 — Độ nhạy siêu tham số

**Trên slide**
- 3–4 biểu đồ nhỏ (accuracy theo giá trị siêu tham số):
  - Mask ratio
  - p_c và p_τ
  - λ (trọng số của L_d)
  - γ₁, γ₂ `[nếu kịp]`
- Nhận xét: mô hình ổn định / nhạy với tham số nào `[ĐIỀN]`

**Cần chuẩn bị:** TV7 xuất biểu đồ cùng style, 1 dataset heterophilic + 1 homophilic.

---

### Slide 25 — Trực quan hóa biểu diễn

**Trên slide**
- **t-SNE** biểu diễn node: GraphMAE vs DGMAE (trên Squirrel hoặc Actor), tô màu theo lớp
- **Độ tương đồng các cặp node nối nhau nhưng khác lớp:** GraphMAE vs DGMAE (tái hiện ý Figure 2 / Figure 8 của paper)
- `[nếu kịp]` Độ đồng đều (uniformity) của biểu diễn
- Thông điệp: DGMAE giữ được sự khác biệt → các cụm tách bạch hơn

**Cần chuẩn bị:** Hình t-SNE 2 cột + biểu đồ phân bố độ tương đồng.

---

### Slide 26 — Kết luận

**Trên slide**
- **Đóng góp của paper:**
  - Chỉ ra GMAE bỏ qua sự khác biệt giữa các node → kém trên đồ thị heterophilic
  - Đề xuất tái tạo discrepancy (L_sym·X) + chọn cạnh thích ứng theo attention
  - Vượt các phương pháp tự giám sát SOTA trên 3 tác vụ, 17 dataset
- **Nhóm đã làm được:** implement lại `[ĐIỀN]`, tái hiện `[ĐIỀN]` dataset, kết quả lệch trung bình `[ĐIỀN]`
- **Hạn chế:**
  - Thêm một lượt encoder trên đồ thị không che → tăng chi phí tính toán
  - Thêm nhiều siêu tham số (λ, p_c, p_τ, γ₂)
  - Hạn chế trong bản implement của nhóm `[ĐIỀN]`
- **Hướng phát triển:** `[ĐIỀN]` (vd. đồ thị lớn hơn, kết hợp tái tạo cấu trúc cạnh)

---

### Slide 27 — Cảm ơn & Hỏi đáp

**Trên slide**
- Cảm ơn thầy/cô và các bạn đã lắng nghe
- Code của nhóm: `[ĐIỀN link repo]`
- Tài liệu tham khảo chính:
  - Zheng et al., *Discrepancy-Aware Graph Mask Auto-Encoder*, KDD 2025 (arXiv:2506.19343)
  - Hou et al., *GraphMAE: Self-Supervised Masked Graph Autoencoders*, KDD 2022
  - Veličković et al., *Graph Attention Networks*, ICLR 2018

---

## Phụ lục A — Phân công trả lời câu hỏi

| Câu hỏi dễ gặp | Trả lời chính |
|---|---|
| Vì sao GMAE kém trên heterophilic? Homophily ratio là gì? | TV2 |
| Vì sao dùng SCE thay MSE? Mask ratio chọn thế nào? | TV3 |
| Vì sao L_sym·X biểu diễn sự khác biệt? Vì sao chọn cạnh attention thấp? Projector để làm gì? | TV4 |
| Kết quả lệch paper bao nhiêu, vì sao? | TV5 (phân loại node), TV6 (phân cụm, đồ thị) |
| Bỏ thành phần nào giảm mạnh nhất? | TV7 |
| Code tự viết hay dùng lại? Train mất bao lâu? | TV1 |

## Phụ lục B — Checklist trước khi chốt slide

- [ ] Đối chiếu mọi chỗ `[KIỂM TRA]` với paper (đặc biệt: loss tổng Mục 4.4, protocol Mục 5.3, Eq. 12)
- [ ] Điền hết `[ĐIỀN]`; số của nhóm và số của paper đặt ở cột riêng, ghi rõ nguồn
- [ ] Sơ đồ kiến trúc vẽ lại, phần mới tô màu khác
- [ ] Ký hiệu thống nhất theo bảng ở đầu file
- [ ] Mỗi phần có câu chuyển ở slide cuối (slide 8 → 9 không đổi người; slide 10 → 11 và 17 → 18 đổi người)
- [ ] Mốc thời gian: T−7 nháp từng phần · T−5 ghép · T−3 tập lần 1 (bấm giờ) · T−1 tập lần 2 & chốt
- [ ] Mỗi người tự soạn 3 câu hỏi khó về phần mình kèm câu trả lời
