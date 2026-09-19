# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Hà Huy Nhất
**Nhóm:** MegaLive
**Ngày:** 2026-09-19

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**

> Độ tương tự cosine cao nghĩa là hai vector embedding có hướng gần giống nhau, tức hai đoạn văn bản có ý nghĩa/ngữ cảnh gần nhau dù có thể dùng từ khác nhau. Điểm càng gần 1 thì mức tương đồng càng cao.

**Ví dụ có độ tương tự CAO:**

- Câu A: Sinh viên cần đăng ký học phần trước hạn.
- Câu B: Người học phải hoàn tất việc chọn môn trước deadline.
- Tại sao tương đồng: Hai câu đều nói về việc sinh viên phải hoàn thành đăng ký môn học đúng thời hạn.

**Ví dụ có độ tương tự THẤP:**

- Câu A: Thư viện cho phép mượn sách trong 14 ngày.
- Câu B: Python là một ngôn ngữ lập trình phổ biến.
- Tại sao khác: Hai câu thuộc hai chủ đề khác nhau, một câu về dịch vụ thư viện và một câu về lập trình.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**

> Cosine similarity tập trung vào hướng của vector, phù hợp hơn để so sánh ý nghĩa của text embedding. Khoảng cách Euclid dễ bị ảnh hưởng bởi độ lớn vector, trong khi điều ta cần là mức gần nhau về ngữ nghĩa.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> Công thức: ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23.
> Đáp án: 23 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**

> Khi overlap tăng lên 100, bước nhảy còn 400 ký tự nên số chunk tăng lên 25. Overlap lớn hơn giúp giữ ngữ cảnh giữa hai chunk liền kề, nhưng đổi lại tốn thêm lưu trữ và truy xuất nhiều chunk hơn.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:

> Em dùng regex `(?<=[.!?])\s+` để tách tại khoảng trắng đứng sau dấu kết thúc câu, nhờ đó vẫn giữ lại dấu câu trong chunk. Sau khi tách, em loại bỏ khoảng trắng thừa và gom tối đa `max_sentences_per_chunk` câu vào một chunk. Trường hợp text rỗng trả về danh sách rỗng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:

> Em chia văn bản theo thứ tự separator ưu tiên từ lớn đến nhỏ: đoạn văn, dòng, câu, từ, rồi ký tự. Nếu một phần vẫn dài hơn `chunk_size`, hàm tiếp tục đệ quy với separator nhỏ hơn; sau đó các phần nhỏ được gom lại để tránh tạo quá nhiều chunk vụn. Base case là text rỗng, text đã ngắn hơn `chunk_size`, hoặc không còn separator thì cắt theo kích thước cố định.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:

> Em dùng in-memory store, mỗi document được lưu thành một record gồm `id`, `content`, `metadata` và `embedding`. Khi search, query được embed một lần, sau đó tính dot product giữa query embedding và embedding của từng record. Kết quả được sắp xếp giảm dần theo `score` và lấy top-k.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:

> `search_with_filter` lọc metadata trước rồi mới tính similarity, đúng với mục tiêu giảm ứng viên nhiễu trước khi retrieval. `delete_document` xóa tất cả record có `metadata["doc_id"]` trùng với `doc_id` cần xóa; khi thêm document, em tự gán `doc_id` bằng `doc.id` nếu metadata chưa có.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:

> Agent lấy top-k chunk liên quan từ `EmbeddingStore.search`, đánh số từng chunk rồi ghép vào phần `Context` của prompt. Prompt yêu cầu LLM trả lời dựa trên context được cung cấp và đặt câu hỏi ở cuối. Cách này mô phỏng luồng RAG cơ bản: retrieve trước, generate sau.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
42 collected, 42 passed
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A                                                | Câu B                                                            | Dự đoán | Điểm thực tế | Đúng? |
| ---- | ----------------------------------------------------- | ----------------------------------------------------------------- | ---------- | ---------------- | ------- |
| 1    | Sinh viên cần đăng ký học phần trước hạn.   | Người học phải hoàn tất việc chọn môn trước deadline.  | cao        | 0.8540           | Có     |
| 2    | Thư viện cho phép mượn sách trong 14 ngày.     | Sinh viên có thể vay tài liệu ở thư viện trong hai tuần. | cao        | 0.8637           | Có     |
| 3    | Học phí được thanh toán qua cổng trực tuyến. | Trời hôm nay có mưa lớn vào buổi chiều.                   | thấp      | 0.5814           | Có     |
| 4    | Phúc khảo bài thi cần nộp đơn theo quy định. | Yêu cầu xem lại điểm thi phải gửi biểu mẫu hợp lệ.     | cao        | 0.7998           | Có     |
| 5    | Ký túc xá có quy định giờ ra vào.             | Python là một ngôn ngữ lập trình phổ biến.                | thấp      | 0.5433           | Có     |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

> Kết quả bất ngờ nhất là các cặp khác chủ đề vẫn có điểm khoảng 0.54-0.58 thay vì gần 0 tuyệt đối. Điều này cho thấy embedding thật biểu diễn ngữ nghĩa trong một không gian liên tục: văn bản khác chủ đề vẫn có một mức tương đồng nền vì cùng là câu tiếng Việt tự nhiên, nhưng các cặp cùng nghĩa vẫn cao hơn rõ rệt, khoảng 0.80-0.86.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query)                                                                               | Top-1 Chunk truy xuất được (tóm tắt)                                                                                       | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt)                                                                                                         |
| - | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------ | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Sinh viên hệ Chuẩn phải đóng học phí theo hình thức nào?                             | `hoc-phi` chunk 1: Hệ Chuẩn thu theo tín chỉ, mức thu thay đổi từng năm theo Nghị định của Chính phủ.           | 0.7369       | Có                               | Sinh viên hệ Chuẩn đóng học phí theo tín chỉ; mức thu thay đổi từng năm theo quy định/Nghị định của Chính phủ.          |
| 2 | Khi nào sinh viên bị kỷ luật Cảnh cáo thì Điểm rèn luyện tối đa là bao nhiêu?   | `hoc-bong` chunk 2: phần cách tính điểm rèn luyện, chưa chứa trực tiếp dòng kỷ luật Cảnh cáo.                  | 0.7006       | Có một phần                    | Chunk liên quan trực tiếp nằm ở top-2: kỷ luật Cảnh cáo thì ĐRL tối đa loại Trung bình.                                        |
| 3 | Sinh viên người dân tộc thiểu số thuộc hộ nghèo được miễn giảm học phí ra sao? | `thu-tuc-mien-giam-hoc-phi` chunk 8: nói về nhóm dân tộc thiểu số được giảm 70% trong vùng đặc biệt khó khăn. | 0.8404       | Có một phần                    | Chunk top-2`hoc-phi` nêu đúng hơn: sinh viên dân tộc thiểu số thuộc hộ nghèo/cận nghèo thuộc diện miễn 100% học phí.     |
| 4 | Để đạt điểm rèn luyện loại xuất sắc cần bao nhiêu điểm?                          | `hoc-bong` chunk 3: phân loại điểm rèn luyện, Xuất sắc: 90 - 100 điểm.                                               | 0.7414       | Có                               | Để đạt loại Xuất sắc cần điểm rèn luyện từ 90 đến 100 điểm.                                                                  |
| 5 | Sinh viên khuyết tật có được ưu tiên điểm rèn luyện không?                        | `hoc-bong` chunk 2: phần cách tính điểm rèn luyện, chưa chứa trực tiếp dòng sinh viên khuyết tật.               | 0.7265       | Có một phần                    | Chunk liên quan trực tiếp nằm ở top-2: sinh viên khuyết tật/hoàn cảnh đặc biệt có cơ chế cộng điểm ưu tiên sự nỗ lực. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**

> Chiến lược fixed size dễ triển khai và hoạt động ổn khi dùng embedding thật, nhưng nó có thể đặt đáp án ở top-2/top-3 nếu chunk bị cắt chưa đúng ranh giới ý. Em học được rằng dữ liệu sạch và chiến lược chunking theo cấu trúc nội dung cũng rất quan trọng.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí                                           | Điểm tự đánh giá |
| ---------------------------------------------------- | ---------------------- |
| Khởi động (Warm-up)                               | 5 / 5                  |
| Hướng tiếp cận của tôi (My Approach)           | 10 / 10                |
| Hoàn thiện code (Core Implementation — tests)     | 30 / 30                |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5                  |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10                |
| **Tổng phần cá nhân**                      | **60 / 60**      |
