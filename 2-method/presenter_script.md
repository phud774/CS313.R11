# Kịch bản thuyết trình phương pháp DGMAE

**Phạm vi:** Trang 13–19 của `main.pdf`, tương ứng bảy slide phương pháp. Đây là bản lời thoại đầy đủ để tập nói; khoảng 9–10 phút nếu đọc hết. Để giữ tổng thời lượng phần 2 khoảng 7,5 phút khi hai slide Implement được hoàn thiện, có thể lược ví dụ học giả ở trang 15 và các đoạn giải thích sâu về hai số mũ $\gamma_1,\gamma_2$; vẫn dành khoảng 30–40 giây cho sơ đồ tổng kết ở trang 19. Hai trang Implement (20–21) hiện chưa có nội dung để viết lời thoại cụ thể. Các câu in đậm ở cuối mỗi mục là lời chuyển slide.

## Trang 13 — Sai khác gốc giữa nút và láng giềng

Ở phần trước, chúng ta đã thấy vấn đề của các mô hình GMAE thông thường: khi tái tạo một nút bị che từ láng giềng, mô hình có xu hướng học phần thông tin chung giữa các nút. Trên đồ thị dị tính, một cạnh vẫn có thể nối hai nút rất khác nhau. Nếu chỉ giữ phần chung, biểu diễn của chúng có thể trở nên khó phân biệt. DGMAE bổ sung một mục tiêu khác: học cả **phần khác biệt** giữa nút và vùng lân cận.

[Chỉ công thức.] Với nút $i$, ta lấy đặc trưng gốc $x_i$, rồi trừ đi tổng đặc trưng của các láng giềng đã được chuẩn hóa theo bậc $d_i$ và $d_j$. Kết quả là $x_i^D$. Trực giác là phép tính này làm nổi bật phần đặc trưng của nút không được tín hiệu từ láng giềng giải thích tốt. Đây là một vector sai khác trên đồ thị, không đơn thuần là khoảng cách giữa hai nút. Hệ số theo bậc điều chỉnh đóng góp của các cạnh khi các nút có số láng giềng khác nhau.

Đây là **sai khác gốc**, tính với toàn bộ láng giềng. Bước chọn cạnh ở hai trang sau sẽ tạo mục tiêu dùng cho loss sai khác. Encoder nhận $X$ hoặc phiên bản bị che $\hat X$, chứ không nhận trực tiếp $x_i^D$.

**Công thức vừa rồi tính sai khác với mọi láng giềng. Nhưng không phải cạnh nào cũng mang thông tin khác biệt đáng học như nhau, nên DGMAE có thêm bước chọn cạnh.**

## Trang 14 — Attention thấp làm cạnh dễ được chọn

Ý tưởng của bước này là ưu tiên các cặp nút mà mô hình xem là ít tương đồng. GAT encoder tạo ra trọng số attention $w_{ij}$ cho cạnh nối nút $i$ với nút $j$. Trong cách diễn giải của bài báo, attention thấp là dấu hiệu để ưu tiên xem xét sai khác của cặp nút đó. Đây là **tín hiệu để lấy mẫu**, không phải khẳng định chắc chắn rằng hai nút khác nhãn; quá trình học không dùng nhãn để quyết định cạnh nào được chọn.

[Chỉ công thức xác suất.] Xác suất chọn cạnh là $p_{ij}=\min((1-w_{ij})p_c,p_\tau)$. Khi $w_{ij}$ giảm thì $1-w_{ij}$ tăng, nên cạnh có xu hướng được chọn nhiều hơn. Tham số $p_c$ điều chỉnh mức lấy mẫu chung. Tham số $p_\tau$ đặt trần cho xác suất, tránh để một cạnh được chọn với xác suất quá cao. Ví dụ, ở hình bên phải, cạnh nét đứt có attention thấp hơn nên xác suất được chọn cao hơn cạnh liền, nếu các yếu tố khác như nhau.

Sau khi có xác suất, mô hình lấy mẫu Bernoulli để tạo $m_{ij}$. Có thể hiểu $m_{ij}$ như một công tắc: bằng 1 thì giữ cạnh đó khi tính mục tiêu sai khác, bằng 0 thì bỏ qua. Vì đây là lấy mẫu theo xác suất, attention thấp làm cạnh **dễ được chọn hơn**, chứ không bảo đảm cạnh ấy luôn được chọn ở mọi lượt huấn luyện.

**Ta đã có công tắc $m_{ij}$ cho từng cạnh. Slide tiếp theo cho thấy công tắc đó thay đổi mục tiêu $x_i^D$ như thế nào.**

## Trang 15 — Đích sai khác chỉ dùng cạnh được chọn

[Chỉ vào tổng trong công thức.] Sau bước lấy mẫu, mục tiêu của nút $i$ được tính bằng cách cộng chênh lệch $x_i-x_j$ với từng láng giềng $j$ được chọn. Mỗi chênh lệch được nhân với $m_{ij}$ và chuẩn hóa theo $\sqrt{d_i d_j}$. Đây là cách chuyển từ ý tưởng “nút khác láng giềng” sang một vector cụ thể để dùng trong huấn luyện.

Hai trường hợp dưới công thức giúp đọc nó dễ hơn. Nếu $m_{ij}=0$, toàn bộ phần đóng góp của cạnh $(i,j)$ bằng không: cạnh ấy không tham gia mục tiêu sai khác ở lượt này. Nếu $m_{ij}=1$, chênh lệch đặc trưng của cặp nút được giữ lại, sau khi tính hệ số theo bậc. Chẳng hạn, trong ví dụ mạng học giả ở phần mở đầu, hai học giả có thể hợp tác nhưng nghiên cứu hai lĩnh vực khác nhau. Nếu cạnh giữa họ được chọn, DGMAE có một tín hiệu để giữ lại sự khác biệt về đặc trưng của cặp này, thay vì chỉ học phần ngữ cảnh chung.

Ký hiệu vẫn là $x_i^D$, nhưng cần phân biệt hai bước: trang 13 giới thiệu phép đo ban đầu với mọi láng giềng; công thức ở đây là **mục tiêu sau khi chọn cạnh**. Ta đã xác định được điều mô hình cần học từ đặc trưng gốc. Việc còn lại là tạo ra một đại lượng tương ứng từ các biểu diễn mà mạng học được.

**Để tạo đại lượng đó, DGMAE xử lý đồ thị hai lần bằng cùng một encoder: một lần che đặc trưng và một lần giữ đầy đủ.**

## Trang 16 — Hai nhánh huấn luyện DGMAE

Từ đây ta sẽ xem DGMAE huấn luyện bằng **hai nhánh**. Chúng dùng cùng đồ thị, cùng các cạnh $A$ và cùng encoder $f_E$; điểm khác nhau là đặc trưng đầu vào có bị che hay không. Trên slide này ta chỉ nhìn tổng quan, rồi hai slide tiếp theo sẽ trình bày từng nhánh cùng loss tương ứng.

[Chỉ nhánh trái.] Đặc trưng bị che $\hat X$ đi qua encoder và decoder, cho ra $\hat Z$. Nhánh này học khôi phục đặc trưng gốc của **nút bị che** bằng loss $\mathcal L_f$. [Chỉ nhánh phải.] Đặc trưng đầy đủ $X$ đi qua cùng encoder rồi qua projector MLP để tạo $Z$. Nhánh thứ hai hướng tới bảo toàn sai khác trên **nút không bị che** bằng loss $\mathcal L_d$.

**Ta đi vào nhánh có che trước: nó tạo $\hat Z$ ra sao và dùng $\mathcal L_f$ như thế nào?**

## Trang 17 — Nhánh đồ thị bị che: tái tạo đặc trưng

[Chỉ cột trái.] Ở đầu một epoch, mô hình chọn tập nút $\hat V$ để che. Với nút $i$ thuộc tập này, đặc trưng đưa vào encoder là $\hat x_i=0$. Encoder xử lý $\hat X$ cùng các cạnh $A$ để tạo $\hat H$; decoder GAT dùng ngữ cảnh láng giềng để tạo $\hat Z$. Đặc trưng gốc $x_i$ được giữ lại làm mục tiêu huấn luyện, không đưa vào lượt xử lý bị che.

[Chỉ cột phải.] Loss $\mathcal L_f$ so sánh $\hat z_i$ với $x_i$ bằng độ tương đồng cosine, rồi lấy trung bình trên **các nút bị che**. Khi dự đoán cùng hướng hơn với đặc trưng gốc, cosine tăng và loss giảm. Số mũ $\gamma_1>1$ điều chỉnh mức phạt. Ví dụ, nếu che đặc trưng của nút B, ta dùng $\hat z_B$ để dự đoán lại $x_B$; các nút không bị che không nằm trong tổng của $\mathcal L_f$.

Loss này buộc mô hình khai thác thông tin từ láng giềng. Tuy nhiên, trên đồ thị dị tính, láng giềng có thể mang đặc trưng rất khác nút cần tái tạo. Vì vậy, chỉ tối ưu $\mathcal L_f$ có thể làm mất phần khác biệt mà DGMAE muốn giữ.

**Loss thứ hai, $\mathcal L_d$, bổ sung tín hiệu sai khác đó và được tính trên nhóm nút còn lại.**

## Trang 18 — Nhánh đồ thị không che: giữ sai khác

[Chỉ cột trái.] Lượt thứ hai dùng đặc trưng gốc đầy đủ $X$ và cùng encoder $f_E$ để tạo $H$. Projector MLP hai lớp đưa $H$ thành $Z$ trong cùng không gian đặc trưng với $\hat Z$. Sau đó ta lấy $z_i^D=z_i-\hat z_i$. Cần nhớ $\hat z_i$ đến từ nhánh **có che** ở slide trước, chứ nhánh này không tự tạo được toàn bộ sai khác. Còn $x_i^D$ là mục tiêu sai khác tính từ đặc trưng gốc và các cạnh đã chọn ở trang 15.

[Chỉ cột phải.] Dòng đầu viết sai số của từng nút là $\ell_i^D$: so sánh $z_i^D$ và $x_i^D$ bằng cosine similarity, rồi nâng lên số mũ $\gamma_2>1$. Dòng sau lấy trung bình các sai số này trên **các nút không bị che** $V\setminus\hat V$ để được $\mathcal L_d$. Khi hai vector cùng hướng hơn, cosine tăng và loss giảm. Đây là mục tiêu giúp sai khác học được trong biểu diễn đi theo sai khác từ đặc trưng gốc.

Điểm chính ở đây không phải dự đoán nhãn hay buộc hai nút nối nhau phải khác nhãn. Mục tiêu là giữ thông tin về **chênh lệch đặc trưng** trong biểu diễn học được. Nhờ vậy, khi ngữ cảnh láng giềng làm các nút có xu hướng giống nhau hơn, mô hình vẫn có một tín hiệu để bảo toàn phần khác biệt.

**Loss sai khác này không đứng riêng: DGMAE huấn luyện nó cùng loss tái tạo đặc trưng của mô hình GMAE.**

## Trang 19 — Huấn luyện DGMAE: loss và một epoch

DGMAE có hai nhiệm vụ bổ sung cho nhau. $\mathcal L_f$ khôi phục đặc trưng của **nút bị che** từ ngữ cảnh, còn $\mathcal L_d$ bảo toàn sai khác trên **nút không bị che**. Hai slide trước đã định nghĩa riêng từng loss; ở đây ta chỉ cần nối chúng lại để thấy cách mô hình được huấn luyện.

[Chỉ cột trái.] Loss cuối cùng là $\mathcal L=(1-\lambda)\mathcal L_f+\lambda\mathcal L_d$. Hệ số $\lambda$ quyết định mức ưu tiên tương đối của hai nhiệm vụ: tăng $\lambda$ thì nhánh sai khác có trọng số lớn hơn, còn giảm $\lambda$ thì nhánh khôi phục đặc trưng có trọng số lớn hơn. Ta không cần đọc từng ký hiệu trong công thức; điều quan trọng là hai nhánh cùng cập nhật các tham số dùng chung, đặc biệt là encoder.

[Chỉ sơ đồ ở cột phải, đi từ trên xuống.] Trong một epoch, trước hết mô hình che đặc trưng một số nút để có $\hat X$. Tiếp đó, hai lượt xử lý đồ thị tạo $\hat Z$ từ dữ liệu bị che và $Z$ từ dữ liệu đầy đủ. Attention giúp chọn cạnh để tính mục tiêu $x_i^D$, còn hiệu $Z-\hat Z$ tạo sai khác ngữ cảnh $z_i^D$. Cuối cùng, hai loss $\mathcal L_f$ và $\mathcal L_d$ được kết hợp thành $\mathcal L$ để cập nhật mô hình. Như vậy, DGMAE vừa học cách khôi phục thông tin từ láng giềng, vừa học cách giữ lại sự khác biệt với láng giềng.

**Phần phương pháp đến đây là đủ để hiểu hai nhánh phối hợp ra sao. Tiếp theo sẽ là phần nhóm triển khai và đối chiếu với mô hình trong bài báo.**

**Nguồn phương pháp:** Zheng et al., [*Discrepancy-Aware Graph Mask Auto-Encoder*](https://arxiv.org/html/2506.19343), KDD 2025, mục 4, Eq. (1)–(13).
