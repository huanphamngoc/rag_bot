# rules/ — khai báo nghiệp vụ

Thư mục này là nơi **duy nhất** mô tả một loại tài liệu và các luật áp lên nó. Không có SQL hay
formatter nào trong Python nữa: `crawlerrag/rules/` đọc các file ở đây, sinh câu SELECT, dựng tài liệu
và kiểm tra chất lượng.

```
rules/
  catalog.yaml        schema nào trong Postgres của crawler được rút metadata
  qualify.yaml        cổng chặn câu hỏi của người dùng, trước khi gọi model
  doc_types/
    drug_recall.yaml  một file cho một loại tài liệu (tên file = doc_type)
    cpsc_recall.yaml
```

Sửa file ở đây rồi chạy `rules-check` để đối chiếu với metadata rút từ Postgres; sai tên bảng hay tên
cột là lỗi, thiếu quan hệ cha–con chỉ là cảnh báo. Không cần sửa Python, không cần build lại image
(thư mục được mount vào container).

## Một doc type gồm những gì

| Khối | Việc |
|---|---|
| `source` | bảng cha, khoá (phải bằng `crawl.record_change.record_key`), cột soft-delete, `source_id` của crawler |
| `children` | mỗi mục là một `array_agg` tương quan trên một bảng con |
| `title`, `body`, `url`, `metadata` | văn bản và filter của tài liệu |
| `scd2` | thuộc tính nào đổi thì tạo version mới, thuộc tính nào ghi đè tại chỗ |
| `quality` | luật kiểm tra dòng đã rút, chạy **trước** khi ghi |

## Ngôn ngữ giá trị

Một *value spec* là một trong:

| Khoá | Nghĩa |
|---|---|
| `field: col` | một cột cha hoặc một alias trong `children` |
| `join: [a, b]` | nối các giá trị không rỗng bằng `separator` (mặc định `", "`) |
| `template: "...{col}..."` | điền cột vào mẫu; **rỗng nếu bất kỳ cột nào rỗng** |
| `coalesce: [spec, spec]` | spec đầu tiên dựng ra chuỗi không rỗng |
| `const: "chữ"` | hằng |

Thêm `format:` là `iso` (ngày), `year` (số năm), hoặc `thousands` (`1,234,567`; 0 và NULL thành rỗng).

Mẹo quan trọng: `template` trả về rỗng khi thiếu cột, nên `coalesce` các `template` chính là cách
diễn đạt "nếu không có thì dùng cái này".

## Hai điều phải giữ

1. **Văn bản không được đổi vô ý.** `tests/test_rules_build.py` so từng byte với builder Python cũ.
   Đổi một nhãn trong `body` là đổi `content_hash` của mọi tài liệu → lần `ingest` sau cắt chunk lại
   toàn bộ. Chunk nào có text y hệt vẫn giữ vector, nhưng text mới thì phải nhúng lại và **mất tiền**.
2. **`version:`** chỉ cần tăng khi bạn *muốn* đối chiếu lại toàn bộ. Việc phát hiện thay đổi đã tự
   động: signature của watermark chứa digest của chính file này, nên sửa YAML là lần chạy sau tự
   chuyển sang `full`.

## Luật chất lượng

| `rule` | Tham số | Lỗi khi |
|---|---|---|
| `not_null` | `column` | có dòng NULL (hoặc list rỗng) |
| `unique` | `column` | có giá trị trùng |
| `allowed_values` | `column`, `values` | giá trị không rỗng nằm ngoài danh sách |
| `max_null_fraction` | `column`, `max` | tỷ lệ NULL vượt ngưỡng |
| `min_rows` | `min` | số dòng ít hơn ngưỡng (**chỉ áp dụng cho full load**) |

`severity: error` (mặc định) làm cả lô thất bại và **không** đẩy watermark → lần sau đọc lại đúng cửa
sổ đó. `severity: warn` chỉ ghi vào `ingest.quality_finding` rồi chạy tiếp.
