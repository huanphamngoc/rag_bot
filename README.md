# crawler-rag

Ứng dụng RAG (hỏi đáp có dẫn nguồn) trên dữ liệu thu hồi thuốc (FDA) và hàng tiêu dùng (CPSC) mà dự án crawler
(`E:\Job\crawler`) đã thu thập. Dự án có vector database riêng (Postgres 16 + pgvector) và một pipeline nạp
**tăng dần** dựa trên change log của crawler: mỗi lần chỉ đọc bản ghi đã đổi, chỉ nhúng văn bản đã đổi.

```
app/rules/*.yaml ──► doc type · luật chất lượng · cổng chặn câu hỏi
                     │
crawler Postgres ──(analyst_ro, 1 snapshot)──► ingest graph ──► pgvector ──► chat graph ──► web / CLI
  crawl.record_change = watermark        plan→extract→quality→   rag.document   qualify→condense→
  38 bảng → meta.*                       stage(SCD2)→embed       mỗi version 1 dòng  retrieve→answer
```

- **Luật nghiệp vụ là YAML** (`app/rules/`): một loại tài liệu = một file, không có SQL trong Python.
- **Metadata rút từ Postgres** (`meta.*`): file YAML được đối chiếu với bảng/cột/khoá/quan hệ thật trước khi đọc dữ liệu.
- **`rag.document` là SCD Type 2**: giữ mọi version, truy hồi chỉ đọc version hiện tại.
- **LangGraph** điều phối cả hai luồng; luồng hỏi đáp có bước **qualify** chặn câu hỏi không nên tới model.
- **MLflow tự host** (`:5001`) ghi lại từng câu trả lời: span, token, giá tiền — câu hỏi bị chặn tạo trace không có span model nào.

Thiết kế và bằng chứng: [docs/DESIGN.md](docs/DESIGN.md).

## Điều kiện

- Docker Desktop.
- Stack crawler đang chạy (`E:\Job\crawler`: `docker compose up -d postgres`), Postgres ở cổng 5433, role `analyst_ro`.
- Model: mặc định Vertex AI, dùng đăng nhập `gcloud auth application-default login` của máy (`GCLOUD_CONFIG` trong `.env`).

## Bắt đầu

```bash
cp .env.example .env            # điền VECTORDB_PASSWORD, SOURCE_PG_PASSWORD (= ANALYST_RO_PASSWORD của crawler), GCLOUD_CONFIG
docker compose up -d vectordb
docker compose run --rm app migrate
docker compose run --rm app init           # thăm dò model 1 lần, cố định vector(1536), tạo HNSW
docker compose run --rm app catalog        # rút metadata từ DB crawler (chỉ đọc)
docker compose run --rm app rules-check    # đối chiếu rules/*.yaml với metadata đó
docker compose run --rm app plan           # chỉ đọc: chế độ, cửa sổ change_id, số chunk + ký tự sẽ nhúng
docker compose run --rm app ingest         # nạp (lần đầu: toàn bộ; sau đó: tăng dần)
docker compose up -d web                   # http://localhost:8089
```

## Hằng ngày

| Việc | Lệnh |
|---|---|
| Nạp phần mới sau mỗi lần crawler chạy | `docker compose run --rm app ingest` |
| Tự nạp mỗi 15 phút | `docker compose --profile scheduler up -d` |
| Xem watermark so với nguồn, các lô gần đây | `docker compose run --rm app status` |
| Đối chiếu lại toàn bộ (sau `crawler rebuild`) | `docker compose run --rm app ingest --full` |
| Chỉ dựng tài liệu, nhúng sau / giới hạn chi phí | `ingest --no-embed`, `embed --max-chunks 5000` |
| Hỏi | `docker compose run --rm app ask "..."`, `app chat`, `app search "..."` |
| Xem trace của các câu trả lời | `docker compose --profile mlflow up -d` → http://localhost:5001 |
| Chấm điểm một câu trả lời lên trace của nó | `docker compose run --rm app feedback 3 good` |
| Test | `docker compose --profile test run --rm test` |

Sau khi sửa `app/rules/`:

| Việc | Lệnh |
|---|---|
| Kiểm luật với schema thật (bảng, cột, khoá, quan hệ) | `docker compose run --rm app rules-check` |
| Rút lại metadata, xem hoặc xuất ra YAML | `app catalog`, `app catalog --export catalog.yaml` |
| Xem các version của một tài liệu | `app history "drug_recall:D-0853-2026"` |
| Thử cổng chặn câu hỏi, không gọi model | `app qualify "how many recalls in 2026?"` |

Thư mục `app/rules/` được mount vào container nên **không cần build lại image** khi sửa luật. Sửa file nào ảnh
hưởng tới văn bản tài liệu thì lần `ingest` sau tự chuyển sang chế độ `full` (chữ ký watermark chứa digest của
file) — nhưng chỉ cắt lại chunk của tài liệu nào thật sự đổi.

## Tracing bằng MLflow

```bash
# MLFLOW_DB_PASSWORD trong .env (bất kỳ chuỗi nào), rồi:
docker compose --profile mlflow up -d --build     # UI: http://localhost:5001
# MLFLOW_TRACKING_URI=http://mlflow:5000 trong .env để bật tracing, rồi khởi động lại web
docker compose up -d web
```

MLflow 3.16.1 tự host, trace nằm trong database `mlflow` riêng bên trong container `vectordb`. Cổng 5001 vì 5000
đã bị MLflow của dự án crawler chiếm. Telemetry của MLflow bị tắt.

Mỗi câu trả lời là một trace: `rag_ask` → `qualify_question` (GUARDRAIL) → `hybrid_retrieve` (+ embed/vector/text)
→ `generate_answer`, kèm token và **giá tiền** do server tự tính. Câu hỏi bị cổng qualify chặn tạo trace **chỉ có
2 span**, không có span model nào — đó là cách trả lời câu "câu hỏi này có tốn tiền không" bằng chính trace. Tag
`qualify_decision` và `qualify_rule` tìm kiếm được trong danh sách trace.

## Duyệt trước khi trả tiền nhúng

Nhúng là bước duy nhất tốn tiền. `INGEST_EMBED_APPROVAL_CHUNKS=50` trong `.env` nghĩa là: khi một lần
nạp phải nhúng hơn 50 đoạn mới, nó **dừng lại và chờ bạn** thay vì tự trả tiền.

```
⏸  Dừng trước bước nhúng: 132 đoạn mới (144,251 ký tự) vượt ngưỡng 50.
   Tài liệu và watermark đã được ghi; chỉ bước nhúng đang chờ bạn.
   Duyệt:   crawlerrag approve 4d07233cf13e4e459d7f10dc443bd615
   Từ chối: crawlerrag approve 4d07233cf13e4e459d7f10dc443bd615 --no
```

Lúc dừng, tài liệu và watermark **đã ghi xong** — chờ bao lâu cũng không mất việc đã làm, và lần chạy
sau không đọc lại cửa sổ đó. Số in ra là số sẽ bị trừ tiền (văn bản trùng nhau chỉ mua một lần). Từ
chối không phải lỗi: các đoạn vẫn nằm chờ và `crawlerrag embed` nhúng chúng khi bạn muốn.

Đặt `0` để tắt. Cổng này cần `INGEST_GRAPH_CHECKPOINT=true` (mặc định), vì một lần dừng không lưu được
thì không trả lời được.

## Khi câu hỏi chưa rõ, hệ thống hỏi lại

Gõ `insulin` — một chủ đề, chưa phải câu hỏi — và lượt đó **tạm dừng** thay vì bị từ chối:

```
That is a subject, not a question yet. Here is what the records actually say about it.

  • Why did Eli Lilly & Company recall a drug?      FDA drug recall D-0445-2024 - Eli Lilly & Company
  • Why did Novo Nordisk Inc recall a drug?         FDA drug recall D-0615-2021 - Novo Nordisk Inc
```

Bấm một câu (hoặc gõ câu của bạn) thì **chính lượt đang chờ** chạy tiếp — không phải lượt mới. Các câu
mẫu được dựng từ bản ghi **thật trong index** theo từ bạn vừa gõ, nên chúng luôn trả lời được và không
bao giờ lệch khi dữ liệu đổi. Sinh chúng chỉ dùng từ khoá, không gọi nhúng, nên hỏi lại miễn phí.

Chỉ câu **mơ hồ** được hỏi lại. Câu ngoài phạm vi, câu tiêm prompt, hay câu xin giúp làm điều có hại thì
bị từ chối và đóng lại — mời sửa lại một câu hỏi về cách chế tạo vũ khí là mời thử lại.

Tắt bằng `RAG_CHAT_CLARIFY=false` trong `.env` (cần `docker compose up -d web` để có hiệu lực).

## Trả lời chảy dần trên trang web

Trang web gọi `POST /api/ask/stream` và hiện câu trả lời **trong lúc model viết**, rồi dựng lại bong
bóng hoàn chỉnh (trích dẫn, nguồn, token, link trace) khi xong. Hội thoại vẫn được giữ nên hỏi tiếp
ngay trong cùng session.

Câu hỏi bị cổng qualify chặn không có mảnh nào chảy ra — vì không có lần gọi model nào. Provider không
hỗ trợ stream (ollama, các endpoint kiểu OpenAI) vẫn chạy bình thường, chỉ không có hiệu ứng gõ chữ.

## Nâng cấp một index đã có sẵn

V005 biến `rag.document` thành SCD Type 2 và chuyển `rag.chunk` từ `doc_id` sang `doc_sk`. Nó **không** ghi lại
dòng chunk nào, nên vector (và các entry HNSW) được giữ nguyên.

**Migrate trước khi build lại image.** Image mới chạy với database cũ sẽ trả HTTP 500
(`column "is_current" does not exist`).

```bash
# đường lùi: một bản sao chưa migrate, mất vài giây, không đụng DB gốc
docker exec crawler-rag-vectordb psql -U rag -d postgres -c "CREATE DATABASE rag_pre_scd2 TEMPLATE rag"
docker compose run --rm app migrate
docker compose run --rm app plan         # kỳ vọng: mode full, 0 chunk cần nhúng
docker compose run --rm app ingest
```

Đã chạy thật trên index 793 MB (xem mục kết quả bên dưới): 35.796/35.796 vector giữ nguyên, lần `ingest` đầu tốn
**0 request nhúng** trong 9,1 s.

## Kết quả (2026-10-04, dữ liệu thật của crawler, máy này)

**Nạp lần đầu (full)**

| Bước | Kết quả |
|---|---|
| `plan` (chỉ đọc) | drug_recall 17.937 key → 18.771 chunk cần nhúng, 17,5 triệu ký tự; cpsc_recall 10.002 key → 16.931 chunk, 18,5 triệu ký tự |
| `ingest --no-embed` | 27.939 tài liệu, 35.746 chunk, 199.310 từ khoá trong 29 s (lô 1: 18,0 s, lô 2: 11,1 s) |
| So với index cũ của crawler | 27.939/27.939 tài liệu và 35.746/35.746 chunk trùng hash: văn bản giống từng byte |
| Nhúng, lần 1 | dừng sau 11.136 text / 174 request / 19 phút vì quota mỗi phút của Vertex AI (HTTP 429), phần đã nhúng được giữ |
| Nhúng, lần 2 | 24.616 text / 385 request / 41 phút, `succeeded`. Có 184 phản hồi 429 tạm thời, retry của provider xử lý hết; cơ chế chờ quota mới thêm (`quota_waits`) không phải dùng lần nào |
| Tổng phần nhúng | 35.752 text khác nhau, 36,0 triệu ký tự, **11.092.546 token** (Vertex báo), 0 text bị cắt; ≈ **$1,66** theo giá $0,00015 / 1.000 token |
| Kích thước | vector DB 793 MB, trong đó HNSW 279 MB |

**Nạp tăng dần (real)**

1. Crawler chạy `crawl openfda_enforcement --mode incremental` (run #20): 2 request HTTP, 123 bản ghi → 50 mới,
   7 đổi, 66 giữ nguyên; change log thêm 57 dòng.
2. `plan`: cửa sổ change_id (155.804, 1.528.261], 57 key → 50 tài liệu mới, 6 đổi, **1 không đổi** (bản ghi nguồn đổi
   nhưng văn bản tài liệu không đổi), 56 chunk cần nhúng.
3. `ingest`: đúng như plan, lô mất **0,077 s** (lô full: 18 s); cpsc_recall `nothing`.
4. `ingest` lần nữa: cả hai `nothing`, 0 lần gọi model, 2,2 s tính cả khởi động container.

**Hỏi đáp** (web `:8089`, Vertex `gemini-2.5-flash`):

- "Why did Pfizer recall a drug?" + bộ lọc `year=2026` → nguồn [1] là `D-0853-2026`, bản ghi vào bằng lô tăng dần,
  được cả hai retriever tìm ra; trả lời "Lack of Assurance of Sterility [1]" (3,9 s; 4.073 / 147 token).
- Cùng câu hỏi viết "in September 2026", **không** có bộ lọc → không tìm ra bản ghi đó: các recall cũ của Pfizer xếp
  trên, còn ngày trong văn bản ở dạng `2026-09-04`. Model trả lời đúng là các đoạn truy hồi không chứa thông tin, không
  bịa. Câu hỏi theo thời gian hiện cần bộ lọc `year` (docs/DESIGN.md mục 11).

## Kết quả (2026-10-05, sau khi chuyển sang YAML + SCD2 + LangGraph)

Chạy **chỉ đọc** trên DB crawler thật và index đang chạy — không ghi, không migrate:

| Đo được | Kết quả |
|---|---|
| Catalog của `crawl`, `drug`, `retail` | 38 bảng, 383 cột, 25 quan hệ (tất cả là foreign key khai báo thật) |
| `rules-check` trên catalog đó | **0 lỗi, 0 cảnh báo** |
| Dựng lại toàn bộ tài liệu từ YAML rồi so `content_hash` với index đang chạy | **27.989/27.989 giống nhau**, 0 khác → chuyển sang YAML không tốn một đồng nhúng nào |
| Index hiện tại | 27.989 tài liệu, 35.796 chunk, 35.796 vector, 793 MB, đang ở V004 |
| Cổng qualify trên 10 câu hỏi thật | phân loại đúng cả 10 (bảng đầy đủ trong docs/DESIGN.md mục 9) |

Phát hiện khi thử câu hỏi thật: tiếng Việt **không dấu** ("Tong so vu thu hoi ... la bao nhieu?") lúc đầu bị xếp
là ngoài phạm vi. So khớp luật giờ bỏ dấu ở cả hai phía, câu hỏi người dùng gõ thì không bị đổi.

### Migration và tracing, chạy thật trên index 793 MB

| Bước | Kết quả |
|---|---|
| `migrate` (sau khi tạo bản sao `rag_pre_scd2` làm đường lùi) | 27.989 tài liệu đều `is_current` version 1; **35.796/35.796 vector giữ nguyên**; 793 MB → 905 MB (index mới) |
| `plan` | mode `full`, **0 mới, 0 đổi, 27.989 giữ, 0 chunk cần nhúng** (7,9 s) |
| `ingest` lần đầu | đúng như `plan`, **0 request nhúng** (9,1 s) |
| `ingest` lần sau | `incremental` / `nothing`, 0 lần gọi model |
| Cổng chất lượng | tìm ra 1/17.987 dòng có `classification = "Not Yet Classified"` — giá trị FDA dùng thật; luật đã được sửa để nhận nó |
| Sửa luật chất lượng rồi chạy lại | quay về `incremental` (digest không đổi) và có hiệu lực **không cần build lại image** |
| Trang web | `/api/ask` trả lời đúng `D-0853-2026` kèm `trace_url` |

**Trace đo được** (Vertex AI thật, MLflow tự tính giá):

| Câu hỏi | Quyết định | Span | Thời gian | Giá |
|---|---|---|---|---|
| Why did Pfizer recall a drug in 2026? | `pass` | 7 | 9,4 s | $0,00289 |
| What hazard did CPSC report for the stroller? | `pass` | 7 | 7,3 s | $0,00293 |
| How many drug recalls were there in 2026? | `needs_sql` | **2** | 0,49 s | **0** |
| Ignore all previous instructions… | `reject` | **2** | 0,51 s | **0** |
| What is the capital of France? | `reject` | **2** | 0,70 s | **0** |

**Test:** 670 đạt (`docker compose --profile test run --rm test`), trước đó 144.
