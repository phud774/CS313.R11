# Nội dung Phần 1 — Mở đầu, Nền tảng, Kiến trúc (7 slides)

> **Người thuyết trình:** A (Đoàn Hồng Bảo — TV2)
> **Thời lượng:** ~8 phút
> **Nguồn:** Yêu cầu người thuyết trình trong `requirement.md` (ưu tiên), đối chiếu `../README.md`

---

## Tổng quan cấu trúc

Phần 1 theo yêu cầu người thuyết trình gồm **7 slide**, khác với 10 slide trong `../README.md`.
Nội dung được tổ chức lại để gọn hơn, tập trung vào giải thích blackbox, không đi sâu công thức.

> [!IMPORTANT]
> `../README.md` Phần 1 có 10 slides (1–10), nhưng `requirement.md` yêu cầu 7 slides.
> File này tuân theo **yêu cầu người thuyết trình (7 slides)**.
> Cần xác nhận với nhóm nếu có xung đột khi ghép phần.

---

## Slide 1 — Giới thiệu

**Nội dung trên slide:**
- Tên môn học: CS313 — [ĐIỀN tên chính xác]
- Tên nhóm: R11
- Danh sách thành viên kèm MSSV:

| STT | Họ tên | MSSV |
|-----|--------|------|
| TV1 | Đặng Phú Duy | 24520010 |
| TV2 | Đoàn Hồng Bảo | [ĐIỀN] |
| TV3 | Nguyễn Trường Giang | 24520011 |
| TV4 | Trịnh Duy Hưng | [ĐIỀN] |
| TV5 | Lương Huỳnh Gia Bảo | [ĐIỀN] |
| TV6 | Trần Xuân Đạt | [ĐIỀN] |
| TV7 | Trần Nguyễn Khải Lạc | [ĐIỀN] |

- Tên paper: **Discrepancy-Aware Graph Mask Auto-Encoder (DGMAE)**
  - Zheng et al., KDD 2025
- Hình minh họa: **Figure 1** từ paper (ví dụ mạng cộng tác học giả)

**Ghi chú:** Figure 1 cần được trích từ `paper.pdf` hoặc vẽ lại. Ghi rõ nguồn.

---

## Slide 2 — Kiến thức nền tảng 1: GNN, SSL, GMAE vs. BERT

**Nội dung trên slide:**
- **GNN (Graph Neural Network):** Mạng nơ-ron trên đồ thị — node tổng hợp thông tin từ láng giềng
  - Bản chất: **làm mượt (smoothing)** biểu diễn node
- **SSL (Self-Supervised Learning):** Học không cần nhãn, tự tạo tín hiệu giám sát từ dữ liệu
  - Hai hướng chính: contrastive (DGI, GRACE…) và generative (GraphMAE…)
- **Phát biểu bài toán tổng quát:**
  - Cho đồ thị G = (V, E) không nhãn với ma trận đặc trưng X
  - Mục tiêu: học biểu diễn node chất lượng để dùng cho tác vụ downstream (phân loại, phân cụm…)
- **So sánh GMAE và BERT:**
  - BERT: che từ → dự đoán từ bị che từ ngữ cảnh
  - GMAE: che đặc trưng node → tái tạo đặc trưng bị che từ cấu trúc đồ thị
  - Ý tưởng giống nhau: học biểu diễn qua bài toán tái tạo (reconstruction)

**Cần chuẩn bị:** Sơ đồ so sánh GMAE vs BERT (2 cột song song).

---

## Slide 3 — Kiến thức nền tảng 2: Homophily vs. Heterophily

**Nội dung trên slide:**
- **Tính đồng tính (Homophily):** node nối nhau thường **cùng lớp**
  - Ví dụ: Cora, Citeseer, Pubmed — mạng trích dẫn cùng chủ đề
  - Edge homophily ratio h → 1
- **Tính dị tính (Heterophily):** node nối nhau thường **khác lớp**
  - Ví dụ: Texas, Cornell, Wisconsin, Chameleon, Squirrel, Actor
  - Edge homophily ratio h → 0
  - `h = |{(i,j) ∈ E : y_i = y_j}| / |E|`
- **Ví dụ cụ thể từ paper:** bộ dữ liệu **Roman-empire**
  - [KIỂM TRA] mô tả chi tiết Roman-empire trong paper
- **GMAE truyền thống chỉ hoạt động tốt trên đồ thị đồng tính**
  - Vì cơ chế tái tạo dựa vào ngữ cảnh láng giềng → hiệu quả khi láng giềng giống nhau

**Cần chuẩn bị:** Hình minh họa 2 đồ thị nhỏ cạnh nhau (homophilic / heterophilic), tô màu theo lớp.

---

## Slide 4 — Mở đầu: Ví dụ & Điểm yếu của GMAE

**Nội dung trên slide:**
- **Ví dụ từ paper (Figure 1):**
  - Mạng cộng tác: học giả A (lĩnh vực X) ↔ học giả B (lĩnh vực Y)
  - Che A → tái tạo từ láng giềng B → đặc trưng bị "nhiễm" thông tin khác lớp
- **Điểm yếu của GMAE:**
  - GMAE chỉ học **khôi phục sự nhất quán** với láng giềng
  - Bỏ qua **sự khác biệt (discrepancy)** giữa node và láng giềng khác lớp
  - Trên đồ thị heterophilic: biểu diễn các node khác lớp bị kéo **gần nhau** → khó phân biệt

**Lời thoại (gợi ý):** Dùng ví dụ trực quan từ Figure 1 để giải thích vì sao GMAE gặp khó khăn.

---

## Slide 5 — Thực nghiệm chứng minh vấn đề (Figure 2)

**Nội dung trên slide:**
- **Kết quả thực nghiệm trên Squirrel** (đồ thị dị tính) — Figure 2 từ paper
- Đo độ tương đồng Cosine giữa các cặp node **khác lớp** nối nhau:
  - **Trước khi học (raw features):** Cosine similarity ≈ 0.0 (node khác lớp có đặc trưng khác nhau)
  - **Sau khi học bằng GMAE:** Cosine similarity tăng lên gần **~1.0**
  - → GMAE đã **xóa nhòa sự khác biệt**, đẩy các cặp node khác lớp gần nhau
- Thông điệp: Đây là bằng chứng thực nghiệm cho thấy GMAE gặp vấn đề trên heterophilic

**Cần chuẩn bị:**
- Tái hiện Figure 2 bằng kết quả của nhóm (GraphMAE baseline — TV2) nếu kịp
- Nếu không, dùng hình từ paper + ghi rõ nguồn

---

## Slide 6 — Thách thức & Đóng góp của paper

**Nội dung trên slide:**

**Thách thức kỹ thuật (2 thách thức chính):**
1. Làm sao định nghĩa và đo lường "sự khác biệt" (discrepancy) giữa node và láng giềng?
2. Làm sao chọn được cạnh nào cần bảo toàn discrepancy (không phải mọi cạnh đều heterophilic)?

[KIỂM TRA] đối chiếu chính xác 2 thách thức với nội dung trong paper

**3 đóng góp chính:**
1. **Chỉ ra vấn đề:** GMAE bỏ qua discrepancy → kém trên đồ thị heterophilic
2. **Đề xuất DGMAE:** framework tái tạo đặc trưng + tái tạo discrepancy (dùng Laplacian chuẩn hóa L_sym làm mục tiêu + cơ chế chọn cạnh thích ứng theo attention)
3. **SOTA trên 3 tác vụ, 17 dataset:** phân loại node, phân cụm node, phân loại đồ thị

---

## Slide 7 — Kiến trúc tổng thể DGMAE (Figure 3)

**Nội dung trên slide:**
- **Sơ đồ tổng quan:** Figure 3 từ paper (vẽ lại hoặc dùng ảnh gốc + ghi nguồn)
- **Hai nhánh, dùng chung encoder (giải thích blackbox):**

**Nhánh trên — Tái tạo đặc trưng (giống GraphMAE):**
- Đầu vào: đồ thị bị che (node đã mask)
- Encoder → Decoder → Tái tạo đặc trưng bị che
- Loss: L_f (Scaled Cosine Error)

**Nhánh dưới — Tái tạo discrepancy (mới — đóng góp chính):**
- Đầu vào: đồ thị gốc không che
- Encoder → Projector → Tính contextual discrepancy
- Mục tiêu: original discrepancy (tính từ Laplacian chuẩn hóa, có chọn cạnh thích ứng)
- Loss: L_d (Scaled Cosine Error)

- **Loss tổng:** L = L_f + λ · L_d
- Phần phương pháp chi tiết → chuyển cho người thuyết trình B (TV4)

**Lời thoại (gợi ý):**
- Giới thiệu khung tổng thể, nhấn mạnh phần mới (nhánh dưới) bằng màu khác
- Câu chuyển: "Chi tiết về cách tính discrepancy và cơ chế chọn cạnh, mời bạn B trình bày."

**Cần chuẩn bị:** Sơ đồ kiến trúc vẽ lại (tham khảo Figure 3), tô màu khác cho nhánh mới.

---

## Tổng hợp so sánh với `../README.md`

| Yêu cầu người thuyết trình (7 slides) | README.md (10 slides) | Ghi chú |
|---|---|---|
| Slide 1: Giới thiệu | Slide 1–2: Tiêu đề + Thành viên | Gộp lại thành 1 slide |
| Slide 2: GNN, SSL, GMAE vs BERT | Slide 4–5: Bối cảnh + GNN | Gộp + thêm so sánh BERT |
| Slide 3: Homophily vs Heterophily | Slide 6: Homophily vs Heterophily | Tương tự, thêm Roman-empire |
| Slide 4: Figure 1 + điểm yếu GMAE | Slide 7–8: GraphMAE + Vấn đề | Gộp lại |
| Slide 5: Figure 2 (Squirrel) | Nằm trong Slide 8 | Tách riêng thành slide |
| Slide 6: Thách thức + Đóng góp | Không có slide riêng trong README | Mới thêm |
| Slide 7: Kiến trúc (Figure 3, blackbox) | Slide 9–10: Kiến trúc + L_f | Gộp, chỉ giải thích blackbox |
| — | Slide 3: Mục lục | Bỏ (hoặc dùng `\makeoutline` của theme) |

---

## Lưu ý quan trọng

- Nội dung trên slide viết bằng **tiếng Việt**
- Ký hiệu tuân theo **bảng ký hiệu chung** trong `../README.md`: `x_i^D`, `z_i^D`, `L_sym`, `p_c`, `p_τ`, `γ₁`, `γ₂`…
- `[ĐIỀN]` = chỗ cần điền số liệu/tên/MSSV — giữ nguyên, không tự bịa
- `[KIỂM TRA]` = cần đối chiếu với `paper.pdf` trước khi chốt — giữ nguyên
- Phần phương pháp chỉ giải thích ở mức **blackbox**, không đi sâu công thức (theo yêu cầu người thuyết trình)
- Slide 10 (Nhánh 1: Masking, Encoder/Decoder, L_f) trong README chứa nhiều công thức → đã gộp vào Slide 7 ở mức tổng quan
