# Báo cáo Day 09 — Trần Thu Huyền - 2A202603274S

## 1. Kết luận trước

Pipeline kết luận **Hold**. Cửa nhận giữ 193/224 ảnh (loại 31 ảnh lỗi, trùng hoặc rò rỉ tập đánh giá); sau mô phỏng duyệt 80 ảnh, còn 34/193 ảnh có cảnh báo (17,6%), cao hơn ngưỡng Hold 10%. Không còn lỗi bắt buộc ở vòng QC cuối; dataset `clean_v1` được xuất ở trạng thái bản nháp, chưa phải bản phát hành.

## 2. Lỗi và phân việc

Số dưới đây là số issue theo mã trong `02_error_list_v1.csv`; các mã cửa nhận loại ảnh được nêu riêng. Tổng pipeline: 224 ảnh đầu vào, 193 ảnh được nhận; danh sách QC vòng một có 36 lỗi bắt buộc và 145 cảnh báo.

| Mã lỗi | Số issue | Ai xử lý | Ghi chú |
|---|---:|---|---|
| `CORRUPT_IMAGE` | 2 | Máy | Loại ảnh không đọc được tại cửa nhận |
| `BAD_SIZE` | 3 | Máy | Loại ảnh nhỏ hơn quy tắc dự án |
| `DUP_IMAGE_NEAR` | 7 | Máy | Loại ảnh trùng gần |
| `DUP_IMAGE_EXACT` | 8 | Máy | Loại ảnh trùng byte |
| `LEAK_TEST` | 11 | Máy | Loại ảnh trùng tập tham chiếu |
| `MISSING_LABEL_FILE` | 6 | Người | Máy ghi `needs_label`, không tự tạo nhãn |
| `BAD_FORMAT` / `BAD_CLASS_ID` / `BOX_ZERO` / `DUP_BOX` | 20 tổng | Máy | Xóa dòng nhãn lỗi |
| `BOX_OUT_OF_IMAGE` | 8 | Máy | Cắt box; nếu không thể giữ box hợp lệ thì xóa dòng |
| `BAD_CLASS_NAME` | 2 | Máy | Chuẩn hóa tên lớp trong `data.yaml` một lần |
| `POSSIBLE_MISSING` / `POSSIBLE_WRONG_CLASS` / `POSSIBLE_LOOSE_BOX` / `POSSIBLE_FALSE_POS` | 121 tổng | Người | Chỉ cảnh báo; ưu tiên theo điểm hàng đợi |

`03_fixes.csv` ghi 6 ảnh cần gán nhãn, 20 dòng bị xóa, 8 box được cắt và một lần chuẩn hóa tên lớp. Hàng đợi có 121 ảnh; 80 ảnh được mô phỏng duyệt. Ảnh `img_0020` đứng đầu với điểm 11.524 do có cảnh báo thiếu vật/nhầm lớp và một dòng đã được máy xóa; sáu ảnh thiếu nhãn nhận điểm `NEEDS_LABEL=10` và đứng ngay sau. Đây là ưu tiên hợp lý vì ảnh thiếu nhãn cần người gán từ đầu.

## 3. Calibration (2 ngưỡng)

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do |
|---|---:|---:|---|---|
| `duplicates.phash_dist` | 2 | 16 | `DUP_IMAGE_NEAR`: 1.00 / 0.50 → 1.00 / 0.83 | Giữ precision 1.00 để tránh loại nhầm ảnh, trong sweep này recall cao nhất ở ngưỡng 16; ngưỡng 20 làm recall giảm và precision của exact duplicate cũng giảm. |
| `second_opinion.conf` | 0.25 | 0.6 | `WRONG_CLASS`: 0.56 / 0.83 → 0.62 / 0.83; `MISSING`: 0.50 / 1.00 → 0.57 / 1.00; `LOOSE_BOX`: 0.29 / 0.40 → 0.20 / 0.20 | Tăng độ tin cậy giúp precision tốt hơn cho hai nhóm chính mà vẫn giữ recall đã đo; cảnh báo box lỏng yếu đi nên cần xem ảnh và không xem đây là phép xác minh nhãn. |

Một giới hạn của calibration: chỉ có 39 ảnh và tập chèn lỗi không đại diện hết lỗi tự nhiên. Ngay tại `phash_dist=16`, recall gần trùng là 0.83 (bỏ sót 1/6 lỗi chèn); với ngưỡng confidence 0.6, recall box lỏng chỉ 0.20. Không nên suy rộng các số này thành mức bắt lỗi chắc chắn trên raw.

## 4. Cửa chặn và điểm mù

Đã chạy kiểm tra với ngân sách 0: kết quả **Rework**, `n_errors=6`, `warn_ratio=0.5596`, `exported=null`; không có dataset được xuất. Điều này cho thấy lỗi bắt buộc còn lại chặn xuất bản.

Điểm mù có bằng chứng từ calibration: (1) check ảnh gần trùng bỏ sót một trong sáu lỗi chèn ở ngưỡng được chọn; (2) check box lỏng chỉ bắt được một trong năm lỗi chèn tại `conf=0.6`. Thêm vào đó, các cảnh báo ngữ nghĩa là ý kiến của một model, không phải nhãn chuẩn; cần người kiểm tra ảnh trước khi chấp nhận sửa.

## 5. Train: raw vs clean

Chưa chạy train. Phần train tùy chọn cần Colab/GPU; do đó chưa có kết quả so sánh raw với clean và chưa thể kết luận chất lượng mô hình.

## 6. Quyết định và ghi chú AI

Quyết định hiện tại là **Hold**. Để tiến tới Release, cần xử lý cảnh báo còn lại và chạy lại pipeline; cần giảm tỷ lệ ảnh còn cảnh báo xuống dưới 10% mà vẫn giữ việc kiểm tra nhãn có ý nghĩa.

- AI được dùng để hỗ trợ triển khai bốn module theo docstring và yêu cầu của lab.
- Cách kiểm: chạy 50 test của bốn module, rồi toàn bộ 79 test và hai lần pipeline (thường và ngân sách 0).
- Lỗi AI phát hiện khi kiểm: code ý kiến thứ hai truyền nhầm cấu trúc tuple của nhãn vào phép tính IoU; đã sửa và chạy lại test.

