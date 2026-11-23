# Yêu cầu chung của nhóm:


## Khung slide — DGMAE: Discrepancy-Aware Graph Mask Auto-Encoder

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


## PHẦN 1 — MỞ ĐẦU, NỀN TẢNG, KIẾN TRÚC
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
| TV1 `Đặng Phú Duy` | Trưởng nhóm · Repo, pipeline, tích hợp code |
| TV2 `Đoàn Hồng Bảo` | Kiến thức nền · Baseline GraphMAE · **Thuyết trình phần 1** |
| TV3 `Nguyễn Trường Giang` | Masking, encoder/decoder, loss L_f |
| TV4 `Trịnh Duy Hưng` | Module discrepancy, chọn cạnh thích ứng, L_d · **Thuyết trình phần 2** |
| TV5 `Lương Huỳnh Gia Bảo` | Thực nghiệm phân loại node · **Thuyết trình phần 3** |
| TV6 `Trần Xuân Đạt` | Thực nghiệm phân cụm + phân loại đồ thị |
| TV7 `Trần Nguyễn Khải Lạc` | Ablation, độ nhạy, trực quan hóa, template slide |

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



# Yêu cầu của người thuyết trình
Yêu cầu chung:
* Viết slide bằng Beamer, trình bày mạch lạc, rõ ràng, ngắn gọn.
* Phải giải thích rõ các khái niệm nền tảng và định nghĩa được bài toán tổng quát trước khi đi vào phần phương pháp.
* Phần phương pháp chỉ giải thích ở mức blackbox, không giải thích lại kiến thức nền tảng và không đi sâu vào công thức toán học.

Slide 1: Giới thiệu
* Tên môn học, tên nhóm, danh sách thành viên (kèm MSSV)
* Tên paper
* Hình minh họa: Figure 1
Slide 2: Kiến thức nền tảng 1
* Giải thích ngắn gọn GNN và SSL.
* Phát biểu bài toán tổng quát: Học tự giám sát trên đồ thị không nhãn
* GMAE và BERT: So sánh cơ chế che từ của BERT với cơ chế che đặc trưng nút của GMAE.
Slide 3: Kiến thức nền tảng 2
* Nói về tính đồng tính  và tính dị tính.
* Lấy ví dụ bộ dữ liệu từ vựng Roman-empire trong paper.
* GMAE truyền thống chỉ hoạt động tốt trên đồ thị đồng tính.
Slide 4: Mở đầu
* Đưa ra ví dụ của paper (Figure 1).
* Chỉ ra điểm yếu của GMAE.
Slide 5: Thực nghiệm chứng minh vấn đề
* Đưa ra kết quả thực nghiệm trên tập dữ liệu dị tính Squirrel (Figure 2).
* Chứng minh rằng GMAE đẩy độ tương đồng Cosine của các cặp nút khác lớp từ ~0.0 lên gần ~1.0.
Slide 6: Thách thức và Đóng góp của paper
* Nêu ngắn gọn 2 thách thức kỹ thuật có trong paper.
* 3 đóng góp chính
Slide 7: Kiến trúc tổng thể của paper
* Ảnh sơ đồ tổng quan (Figure 3).
* Giải thích sơ bộ nhánh trên theo blackbox.
* Giải thích sơ bộ nhánh dưới theo blackbox.
