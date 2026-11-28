# Script thuyết trình — Phần 1: Mở đầu & Nền tảng

File này **không hiển thị trên slide**. Với mỗi slide gồm 2 phần:

- **Nội dung chính** — đúng những gì đang có trên slide (để đối chiếu khi tập).
- **Script thuyết trình** — lời nói khi trình bày, giải thích từng dòng, từng keyword/khái niệm xuất hiện trên slide.

Người trình bày: A (TV2 — Đoàn Hồng Bảo) · ~8 phút cho 10 slide.

---

## Slide 1 — Bài toán thực tế: Dữ liệu dạng đồ thị

### Nội dung chính
- Hình trái (Zhou et al. 2020, Fig. 1): Ảnh (Euclidean) vs. Đồ thị (non-Euclidean).
- **Ví dụ:** Mạng xã hội · Mạng trích dẫn · Phân tử · Giao thông.
- **Bài toán:** Phân loại nút — dự đoán nhãn từng nút.
- **Thách thức:** Nhãn **đắt đỏ**, đồ thị **lớn** → thiếu dữ liệu giám sát.
- **Hệ quả:** Học biểu diễn **không giám sát** → **SSL**.

### Script thuyết trình
Mình mở đầu bằng câu hỏi: dữ liệu dạng đồ thị là gì và vì sao nó khó?

Nhìn hình bên trái. Bên trên là **ảnh** — dữ liệu **Euclidean**: một lưới pixel đều đặn, mỗi pixel có đúng mấy láng giềng cố định (trên, dưới, trái, phải) và có thứ tự. Vì có lưới và có thứ tự nên ta mới trượt được kernel của CNN. Bên dưới là **đồ thị** — dữ liệu **non-Euclidean**: mỗi nút có số láng giềng tùy ý, các láng giềng không có thứ tự, và không có hệ tọa độ chung. Không có lưới thì tích chập thông thường không áp dụng được — đó chính là lý do phải có GNN, sẽ nói ở slide sau.

Ví dụ cụ thể: **mạng xã hội** — nút là người dùng, cạnh là quan hệ bạn bè, đặc trưng nút là hồ sơ, sở thích; **mạng trích dẫn** — nút là bài báo, cạnh là "bài này trích bài kia", đặc trưng là túi từ của tiêu đề hoặc abstract, nhãn là lĩnh vực; **phân tử** — nút là nguyên tử, cạnh là liên kết hóa học; **giao thông** — nút là nút giao, cạnh là đoạn đường.

Bài toán con mà paper quan tâm là **phân loại nút**: gán nhãn cho từng nút. Nghe đơn giản, nhưng có hai nút thắt. Thứ nhất, **nhãn đắt đỏ**: gán nhãn phải có chuyên gia, tốn thời gian và tiền. Thứ hai, **đồ thị lớn**: hàng triệu nút, hàng chục triệu cạnh — không ai gán tay nổi. Hệ quả là ta rơi vào tình trạng **thiếu dữ liệu giám sát**.

Vì vậy hướng đi là **học biểu diễn không giám sát**: thay vì học từ nhãn, ta tự sinh tín hiệu giám sát ngay từ dữ liệu — gọi là **SSL, Self-Supervised Learning (học tự giám sát)**. DGMAE trong bài này là một phương pháp SSL cho đồ thị, nên trước khi vào phương pháp, mình sẽ nói rõ SSL vận hành thế nào.

---

## Slide 2 — Bài toán Đồ thị & Học tự giám sát (SSL)

### Nội dung chính
- Hình TikZ: 5 nút $v_1 \dots v_5$, hai màu nút = hai nhãn; $G=(V,E)$, $X \in \mathbb{R}^{|V|\times d}$.
- **Đầu vào:** Đồ thị $G=(V,E)$ kèm đặc trưng nút $X$.
- **Mục tiêu:** Học biểu diễn $X \xrightarrow{\text{học}} H \in \mathbb{R}^{|V|\times d'}$.
- **Ứng dụng:** Phân loại nút · Phân cụm nút.

### Script thuyết trình
Hình bên trái là một đồ thị nhỏ: 5 nút $v_1$ đến $v_5$ có cạnh nối; mình tô hai màu để ý là các nút thuộc hai nhãn khác nhau — nhưng lưu ý, trên hình chỉ *một phần* nút được biết nhãn, đó mới đúng tình huống thực tế.

Ký hiệu dưới hình: **$G=(V,E)$** — $V$ là tập nút, $|V|$ là số nút, $E$ là tập cạnh, và $(i,j)\in E$ nghĩa là nút $i$ nối với nút $j$. **$X \in \mathbb{R}^{|V|\times d}$** là ma trận đặc trưng: mỗi **hàng** là vector $d$ chiều của một nút, $d$ là số đặc trưng — ví dụ số từ trong túi từ. Nói cách khác, đồ thị = cấu trúc $G$ + đặc trưng $X$, và **không có nhãn $y$** — đây chính là điểm xuất phát của bài toán SSL.

Mục tiêu: từ $X$, **học** ra $H \in \mathbb{R}^{|V|\times d'}$ — mỗi nút có một vector embedding $d'$ chiều, thường $d' \ll d$. Chữ "học" ở đây nghĩa là tối ưu tham số của encoder sao cho $H$ vẫn giữ thông tin hữu ích nhưng gọn và sạch hơn $X$: giảm chiều, khử nhiễu, và kéo các nút cùng vai trò/ngữ cảnh lại gần nhau trong không gian vector.

Có $H$ rồi thì dùng cho hai việc: **phân loại nút** — gắn một bộ phân loại nhẹ lên trên $H$, thường chỉ cần logistic regression, và chỉ cần rất ít nhãn; **phân cụm nút** — chạy K-means trên các vector $h_i$ mà không cần nhãn nào cả.

Còn chữ **SSL** ở tiêu đề: tín hiệu giám sát được tự tạo từ $X$ và cấu trúc đồ thị, chứ không lấy từ nhãn. Tạo bằng cách nào thì các slide sau sẽ trả lời.

---

## Slide 3 — GNN: Từ láng giềng đến node embedding

### Nội dung chính
- Hình (Zhou et al. 2020, Fig. 2): Input → GNN Layer → … → Output (Node/Edge/Graph Embedding) → Loss Function (Training Setting × Task).
- GNN gom thông tin từ láng giềng để tạo embedding cho từng nút.

### Script thuyết trình
Slide này trả lời câu "GNN làm gì", đọc theo đúng bốn khối của hình, từ trái sang phải.

**Input**: đồ thị gốc — các nút xanh và cạnh, tức là cấu trúc (ma trận kề $A$) cùng đặc trưng $X$.

**GNN Layer**: một lớp gồm hai bước — **message passing** (mỗi nút gom đặc trưng của các láng giềng) và **update** (cập nhật biểu diễn của chính nó). Dấu "..." nghĩa là ta xếp chồng $L$ lớp như vậy: sau $L$ lớp, vector $h_i$ của nút $i$ đã gom ngữ cảnh trong bán kính **$L$-hop**. Càng nhiều lớp thì tầm nhìn càng xa, nhưng cũng dễ bị **oversmoothing** — làm mượt quá mức khiến mọi nút hóa giống nhau. Đường **đứt nét** phía trên là nhánh tắt: cấu trúc gốc vẫn được đưa thẳng tới bước loss, tức loss có thể ràng buộc theo cấu trúc chứ không chỉ theo embedding.

**Output**: có ba loại. **Node embedding** — một vector cho mỗi nút $h_i$, và DGMAE dùng đúng loại này. **Edge embedding** — vector cho mỗi cạnh, dùng cho dự đoán liên kết. **Graph embedding** — một vector cho cả đồ thị, dùng cho phân loại đồ thị.

**Loss Function**: gồm hai chiều. *Training Setting* — **supervised** (có nhãn), **semi-supervised** (chỉ một phần nút có nhãn, phổ biến nhất), **unsupervised** (không nhãn) — SSL nằm ở ô unsupervised này. *Task* — **node-level** (phân loại/phân cụm nút), **edge-level** (dự đoán cạnh), **graph-level** (phân loại cả đồ thị).

Ý chính cần nhớ: GNN chỉ là **encoder** sinh ra $h_i$; biểu diễn tốt hay không phụ thuộc vào **loss** đặt lên nó. DGMAE giữ nguyên encoder GNN, chỉ thay loss có nhãn bằng loss tự giám sát.

Và lưu ý để nối sang phần sau: chính cơ chế gom láng giềng khiến GNN **làm mượt** biểu diễn các nút kề nhau — trên đồ thị dị tính, đây là gốc rễ của vấn đề.

---

## Slide 4 — Bức tranh Graph SSL: Contrastive vs. Generative

### Nội dung chính
| Tiêu chí | Contrastive | Generative |
|---|---|---|
| Ý tưởng | Tạo 2 góc nhìn nhiễu, kéo cặp dương / đẩy cặp âm | Mã hóa đồ thị và tái tạo lại cấu trúc / đặc trưng |
| Đại diện | DGI, GRACE, BGRL | GAE, MaskGAE, **GraphMAE** |
| Ưu điểm | Học biểu diễn phân biệt tốt | Đơn giản, không cần augmentation phức tạp hay negative sampling |
| Nhược điểm | Phụ thuộc phép tăng cường, tốn chi phí mẫu âm | Phụ thuộc mạnh vào ngữ cảnh láng giềng để tái tạo |

### Script thuyết trình
Graph SSL có hai hướng chính; bảng này so sánh theo bốn tiêu chí, mình đọc ngang từng hàng.

**Ý tưởng.** Contrastive: **tạo 2 góc nhìn nhiễu** — gọi là **augmentation**, ví dụ bỏ bớt nút, bỏ bớt cạnh, đổi đặc trưng, để sinh ra hai phiên bản của cùng một đồ thị. Rồi **kéo cặp dương** (hai góc nhìn của *cùng một nút* — phải lại gần nhau) và **đẩy cặp âm** (hai nút *khác nhau* — phải ra xa nhau). Generative thì khác: **mã hóa đồ thị** thành embedding rồi **tái tạo lại** cấu trúc hoặc đặc trưng đã bị che. Điểm mấu chốt: tín hiệu giám sát của generative chính là **đầu vào gốc**, nên không cần nhãn.

**Đại diện.** Contrastive: **DGI** (tối đa hóa tương hỗ giữa nút và đồ thị), **GRACE** (hai góc nhìn bằng cách bỏ cạnh/đặc trưng), **BGRL** (bỏ negative, dùng teacher–student để đỡ tốn). Generative: **GAE** (tái tạo ma trận kề), **MaskGAE** (che cạnh/đường đi), và **GraphMAE** — nhân vật chính của buổi này.

**Ưu điểm.** Contrastive học biểu diễn phân biệt tốt, nhưng phụ thuộc vào việc thiết kế augmentation và **tốn chi phí mẫu âm**: mỗi bước phải lấy mẫu nhiều negative nên tốn bộ nhớ và thời gian. Generative ngược lại: **không cần augmentation phức tạp, không cần negative sampling** — đơn giản, ổn định, dễ huấn luyện.

**Nhược điểm.** Của generative là **phụ thuộc mạnh vào ngữ cảnh láng giềng**: muốn tái tạo nút bị che thì phải dựa vào các nút kề. Nếu nút kề khác nhãn — tức đồ thị dị tính — thì thông tin dùng để tái tạo là *sai*. Đây chính là khe hở mà DGMAE khai thác, và cũng là lý do phần còn lại của bài tập trung vào nhánh generative.

---

## Slide 5 — Generative methods

### Nội dung chính
- Hình 2 nhánh: GAE (trên) và GraphMAE (dưới).
- **GAE — nhánh trên:** Tái tạo **cạnh**: $A \rightarrow \hat{A}$; Encoder GNN → decoder MLP.
- **GraphMAE — nhánh dưới:** Che đặc trưng nút: $X \rightarrow \hat{X}$; Tái tạo **đặc trưng**, loss **SCE**.

### Script thuyết trình
Hình có hai nhánh: nhánh trên là **GAE**, nhánh dưới là **GraphMAE**. Cả hai đều generative: encoder → decoder → tái tạo, chỉ khác *tái tạo cái gì*.

**GAE — tái tạo cạnh, $A \rightarrow \hat{A}$.** $A$ là **ma trận kề**: $A_{ij}=1$ nếu có cạnh giữa $i$ và $j$. $\hat{A}$ là ma trận kề *dự đoán*. Loss so $\hat{A}$ với $A$, gọi là **link reconstruction** — đoán xem cạnh nào tồn tại; vì đầu ra là xác suất nên thường dùng cross-entropy. Kiến trúc: **encoder GNN** nén mỗi nút thành $h_i$, rồi **decoder MLP** (hoặc đơn giản là tích vô hướng $h_ih_j^\top$) dựng lại cạnh từ cặp embedding.

**GraphMAE — che đặc trưng nút, $X \rightarrow \hat{X}$.** Ta che ngẫu nhiên một tỉ lệ nút (**mask ratio**), thay đặc trưng của nút bị che bằng một **mask token** học được; $\hat{X}$ là đặc trưng được tái tạo. Điểm khác GAE: GraphMAE **không đoán cạnh** mà đoán lại vector đặc trưng gốc của nút bị che, dựa vào các láng giềng *không bị che*.

**Loss SCE** — *Scaled Cosine Error*, dạng $(1-\cos(\hat{x}_i, x_i))^{\gamma}$. Dùng cosine thay vì MSE vì cosine bớt nhạy với độ lớn đặc trưng, giúp huấn luyện ổn định hơn.

**Mục đích chung của cả hai:** encoder chỉ để **sinh embedding**. Huấn luyện xong thì bỏ decoder, đem embedding sang task khác như phân loại hoặc phân cụm — đó là ý nghĩa của pretrain SSL.

**Chuyển ý:** GraphMAE tái tạo đặc trưng nút *hoàn toàn dựa vào láng giềng*. Nếu láng giềng khác nhãn thì nó học sai — điểm yếu này sẽ được phân tích ở các slide tiếp theo.

---

## Slide 6 — Tính đồng tính vs. Tính dị tính: Định nghĩa

### Nội dung chính
- **Homophily (Đồng tính):** phần lớn cạnh nối nút **cùng nhãn** ($y_i = y_j$); tỉ lệ đồng tính $h$ **cao**. Hình: 5 cạnh liền, 1 cạnh đứt → $h = 5/6$.
- **Heterophily (Dị tính):** phần lớn cạnh nối nút **khác nhãn** ($y_i \neq y_j$); $h$ **thấp**. Hình: 1 cạnh liền, 5 cạnh đứt → $h = 1/6$.
- Nét liền: cùng nhãn · Nét đứt: khác nhãn.
- Công thức: $h = \dfrac{|\{(i,j) \in E : y_i = y_j\}|}{|E|}$.

### Script thuyết trình
Slide này chỉ **định nghĩa** hai khái niệm; vì sao chúng gây khó cho GNN là nội dung của slide sau.

Công thức ở dưới là trung tâm: **$h$** là tỉ lệ đồng tính của đồ thị. Tử số là số cạnh nối hai nút **cùng nhãn**; mẫu số là tổng số cạnh $|E|$. Trong đó $(i,j) \in E$ nghĩa là cạnh giữa $i$ và $j$, còn $y_i$ là nhãn của nút $i$. Vậy $h$ luôn nằm trong $[0,1]$.

**Homophily — đồng tính:** phần lớn cạnh nối hai nút **cùng nhãn**, $y_i = y_j$, tức $h$ cao, gần 1. Trực giác là "vật lý gần vật lý", "AI gần AI" — đúng như kỳ vọng. Kiểm chứng bằng hình trái: đếm được 6 cạnh, chỉ có 1 cạnh nối $A$ với $B$ khác nhãn, nên $h = 5/6$.

**Heterophily — dị tính:** ngược lại, phần lớn cạnh nối hai nút **khác nhãn**, $y_i \neq y_j$, nên $h$ thấp, gần 0. Cạnh ở đây nối các *kiểu khác nhau* — ví dụ một tác giả AI viết chung bài với một tác giả Vật lý. Hình phải: 6 cạnh, chỉ 1 cạnh $A$–$A$ cùng nhãn, nên $h = 1/6$.

**Quy ước hình:** màu nút là nhãn; **nét liền** là cạnh cùng nhãn; **nét đứt** là cạnh khác nhãn.

Ý cần nhớ: $h$ chỉ là một con số thống kê của đồ thị. Nhưng hệ quả rất lớn — trong đồ thị dị tính, hai nút nối nhau **không** có nghĩa là giống nhau, và đó là chỗ mọi phương pháp dựa vào láng giềng sẽ vấp.

---

## Slide 7 — Vấn đề: Điểm yếu của GMAE trên đồ thị dị tính

### Nội dung chính
- Hình `example.pdf`: A (AI) ↔ C (AI), B (Vật lý). Che A → B gây **nhiễu**.
- **Điểm yếu cốt lõi:** Thiên về **nhất quán (Consistency)** · Thiếu **sai khác (Discrepancy)** · Kéo nút khác lớp lại **gần nhau**.

### Script thuyết trình
Đây là ví dụ định nghĩa nên toàn bộ động lực của paper.

Hình bên trái: nút **A** và **C** cùng nhãn AI, nút **B** nhãn Vật lý; A nối với cả C và B — đúng kiểu mạng cộng tác nghiên cứu, hai người hợp tác nhưng khác lĩnh vực.

Giờ áp GraphMAE: ta che A rồi bắt mô hình tái tạo A từ các láng giềng. Láng giềng gồm C (liên quan) và B (không liên quan). Đặc trưng của B không giúp gì cho A, nên nó bị trộn vào như **nhiễu** — đó là dòng "Che A → B gây nhiễu".

Ba dòng điểm yếu, giải thích từng chữ:

**Thiên về nhất quán (Consistency)** — mô hình chỉ học phần *chung* giữa nút và láng giềng, tức tối thiểu hóa khoảng cách giữa $x_i$ và ngữ cảnh của nó, hay nói cách khác là **làm cho chúng giống nhau**.

**Thiếu sai khác (Discrepancy)** — không có tín hiệu nào buộc mô hình giữ lại phần *khác* giữa nút và láng giềng, nên phần thông tin đó bị mất trong quá trình tái tạo.

**Kéo nút khác lớp lại gần nhau** — hệ quả trực tiếp: embedding của A bị kéo về phía B, biên phân lớp mờ đi, độ chính xác giảm trên đồ thị dị tính.

Và đây là hướng giải: nếu buộc mô hình tái tạo cả **sai khác** $x_i^D = x_i - \sum_{j\in N(i)} \frac{1}{\sqrt{d_i d_j}} x_j$, thì nút khác lớp vẫn tách được. Đó chính là ý tưởng của DGMAE.

---

## Slide 8 — Thực nghiệm chứng minh: Xóa nhòa sự khác biệt

### Nội dung chính
- Hình `fig1.pdf` — Figure 2: Cosine similarity trên Squirrel.
- **Quan sát (cặp nút khác lớp):** **Raw features:** $\approx 0$ · **GraphMAE / MaskGAE:** $\longrightarrow 1$.
- **Kết luận:** Tăng độ tương đồng nút khác lớp · Giảm khả năng phân biệt biểu diễn.

### Script thuyết trình
Slide trước nói bằng lời; slide này là bằng chứng thực nghiệm, lấy từ Figure 2 của paper, chạy trên tập **Squirrel** — một đồ thị dị tính.

Trước hết **cosine similarity**: $\cos(u,v) = \frac{u \cdot v}{\|u\|\,\|v\|}$, nhận giá trị trong $[-1,1]$. Gần 1 nghĩa là hai vector cùng hướng, tức giống nhau; gần 0 nghĩa là gần trực giao, tức không liên quan.

Đối tượng đo là các **cặp nút khác lớp** — cụ thể là cặp nút *kề nhau nhưng khác nhãn*. Trên Squirrel, vì $h$ thấp, loại cặp này rất nhiều, và về lý thuyết chúng *phải* khác nhau.

**Raw features $\approx 0$**: trước khi học, đặc trưng gốc của các cặp này gần trực giao — đúng như bản chất: chúng khác nhau.

**GraphMAE / MaskGAE $\rightarrow 1$**: sau khi học, phân bố cosine dồn mạnh về 1 — hai nút khác lớp bị kéo thành gần như *giống hệt* nhau.

Hai dòng kết luận: thứ nhất, **tăng độ tương đồng nút khác lớp** — đây chính là hiện tượng homogenization/oversmoothing mà slide trước mô tả bằng lời, giờ thấy rõ bằng số. Thứ hai, **giảm khả năng phân biệt biểu diễn** — khi mọi nút cùng hướng, khoảng cách giữa các lớp gần 0, bộ phân loại phía sau không còn ranh giới để học.

Cách đọc hình: trục hoành là giá trị cosine, trục tung là mật độ (ước lượng KDE); đường càng dồn về phía phải thì các cặp càng giống nhau.

---

## Slide 9 — Động lực phát triển: Thách thức & Đóng góp của DGMAE

### Nội dung chính
- **Tư duy mới:** Consistency $\rightarrow$ **Preserve Discrepancy**.
- **Adaptive Selection:** Dùng attention chọn cạnh sai khác.
- **Hiệu năng:** Tốt nhất **16/17** bộ dữ liệu.

### Script thuyết trình
Từ vấn đề vừa chứng minh, paper đề xuất DGMAE với ba đóng góp — đúng ba dòng trên slide.

**Tư duy mới: Consistency $\rightarrow$ Preserve Discrepancy.** Đây là đóng góp khái niệm: đổi mục tiêu học. Thay vì làm nút giống láng giềng (consistency), DGMAE **bảo toàn phần khác biệt** (discrepancy) giữa nút và láng giềng. Nói cách khác là **định nghĩa lại paradigm SSL** cho đồ thị: từ "khôi phục cái chung" sang "giữ lại cái riêng".

**Adaptive Selection — dùng attention chọn cạnh sai khác.** Không phải cạnh nào cũng đáng giữ sai khác như nhau, nên DGMAE dùng **attention**: mỗi cạnh có trọng số chú ý $w_{ij}$; $w_{ij}$ thấp nghĩa là hai nút ít giống nhau, nên xác suất chọn cạnh đó cao — $p_{ij} = \min((1-w_{ij})p_c,\, p_\tau)$ rồi lấy mẫu theo phân phối Bernoulli. Trong đó $p_c$ chỉnh *số lượng* cạnh được lấy, còn $p_\tau$ là chặn trên để tránh lấy quá nhiều làm phá cấu trúc đồ thị. Nhờ vậy mô hình tự thích nghi với đồ thị có tỉ lệ đồng tính khác nhau.

**Hiệu năng: tốt nhất 16/17 bộ dữ liệu.** Thực nghiệm trên 17 benchmark, gồm cả đồng tính lẫn dị tính; DGMAE đứng đầu ở 16 bộ, trên ba loại task: phân loại nút, phân cụm nút và phân loại đồ thị.

Slide tiếp theo mở kiến trúc để xem hai đóng góp đầu được hiện thực hóa bằng hai nhánh loss như thế nào.

---

## Slide 10 — Kiến trúc tổng thể DGMAE (Figure 3)

### Nội dung chính
- Hình `model.pdf` (Figure 3).
- **2 nhánh song song:** Feature Recon. ($\mathcal{L}_f$) + Discrepancy Recon. ($\mathcal{L}_d$).

### Script thuyết trình
Đây là kiến trúc tổng thể, Figure 3. Chỉ cần nắm một câu: **hai nhánh song song, dùng chung một encoder**.

**Dùng chung encoder $f_E$** nghĩa là cả hai nhánh đều đi qua cùng một GNN để lấy embedding; chỉ phần decoder và loss là khác nhau.

**Nhánh trên — Feature Recon., $\mathcal{L}_f$:** che đặc trưng nút $X$, decoder dựng lại $\hat{X}$; $\mathcal{L}_f$ là SCE giữa $\hat{X}$ và $X$. Nhánh này giống GraphMAE — nó học phần **nhất quán** với láng giềng.

**Nhánh dưới — Discrepancy Recon., $\mathcal{L}_d$:** trước hết tính trước **sai khác gốc** $x_i^D = x_i - \sum_{j \in N(i)} \frac{1}{\sqrt{d_i d_j}} x_j$; về mặt toán học việc này tương đương áp Laplacian $L_{sym}$ lên $X$ — tức một **bộ lọc thông cao**, giữ lại phần biến thiên giữa nút và láng giềng. Trên không gian embedding, "sai khác" được lấy giữa đồ thị **bị che** và đồ thị **không bị che**; $\mathcal{L}_d$ là SCE giữa $z_i^D$ và $x_i^D$.

**Dấu "$+$" ở dòng dưới:** tổng có trọng số $\mathcal{L} = \mathcal{L}_f + \lambda \mathcal{L}_d$, với $\lambda$ cân bằng hai nhánh; hai nhánh hỗ trợ nhau trong lúc huấn luyện.

**Đọc ký hiệu trên hình:** $\hat{x}_i$ là đặc trưng tái tạo; $\hat{H}$ là embedding, $\hat{Z}$ là biểu diễn sau decoder của đồ thị *bị che*; $Z$ là embedding của đồ thị *đầy đủ*, dùng để tạo sai khác ngữ cảnh; còn $x_i^D$ đóng vai **nhãn giả** cho nhánh dưới — nó là tín hiệu tự giám sát, không phải nhãn người gán.

**Vì sao phải tách đôi:** nhánh trên giữ thông tin chung để biểu diễn ổn định; nhánh dưới giữ thông tin riêng để biểu diễn phân biệt. Thiếu nhánh dưới thì quay lại bị làm mượt; thiếu nhánh trên thì mất tín hiệu tái tạo. Chi tiết từng nhánh sẽ do phần Phương pháp trình bày.
