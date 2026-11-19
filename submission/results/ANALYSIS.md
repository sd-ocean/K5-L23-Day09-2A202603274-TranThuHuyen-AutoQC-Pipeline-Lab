# Phân tích thí nghiệm raw vs clean

Thiết lập theo các JSON kết quả: `yolo26n.pt`, 20 epoch, batch 16, ảnh 640 px, hai seed (0 và 1), cùng tập test. Các số tổng hợp lấy từ `comparison.json` và từng file `*_seed*.json`.

## 1. Chênh lệch mAP50 và giới hạn kết luận

`clean` đạt mAP50 trung bình 0.7301, `raw` đạt 0.7258; chênh lệch clean − raw là **+0.0043** (bảng làm tròn thành +0.004). Độ lệch chuẩn giữa hai seed lần lượt là 0.0014 (clean) và 0.0002 (raw). Chênh lệch quan sát được lớn hơn độ dao động seed trong lần chạy này, nhưng chỉ có hai seed và một tập test 100 ảnh. Vì vậy đây là tín hiệu nhỏ có lợi cho clean, chưa đủ để khẳng định clean tốt hơn một cách ổn định.

Các metric khác không đồng nhất: mAP50–95 giảm từ 0.5117 xuống 0.5007; precision tăng từ 0.7635 lên 0.8001; recall giảm từ 0.6490 xuống 0.6243. Không nên tóm tắt kết quả thành “clean tốt hơn” chỉ dựa trên mAP50.

## 2. Số ảnh raw thực sự được dùng

Mỗi seed raw thực sự train trên **213 ảnh**. Log Colab ghi rõ `Using 213 train, 40 val images`; bước scan báo `216 images, 19 backgrounds, 11 corrupt` trên 224 file. Các mục corrupt gồm ảnh hỏng, class ID ngoài 0–3, nhãn không hợp lệ và tọa độ box ngoài biên. Ultralytics loại 11 ảnh khỏi train. Log clean ghi `Using 193 train, 40 val images` và `193 images, 3 backgrounds, 0 corrupt`.

Các JSON raw vẫn ghi `train_images=224` vì `scripts/train_compare.py` đếm file ảnh đã materialize trước khi Ultralytics scan nhãn. Đây là số file đầu vào, không phải số ảnh thực tế được loader dùng; báo cáo cần phân biệt hai con số này.

## 3. Rò rỉ ảnh test vào raw

Có **11 issue `LEAK_TEST`** trong `01_intake`/`02_error_list_v1`; ví dụ `img_0013` trùng `test/test_087.jpg`. Các ảnh này có mặt trong raw và do đó có thể làm điểm raw trên test bị thổi phồng. Pipeline loại chúng khỏi clean.

Không thể tính chính xác mức độ thổi phồng từ các kết quả hiện có. Hai seed raw đã train trên bộ có rò rỉ, và không có đối chứng raw được train lại sau khi bỏ đúng 11 ảnh đó. Cách kiểm là tạo biến thể raw đã loại các ảnh `LEAK_TEST`, giữ nguyên model, epoch, seed, val/test và mọi siêu tham số, rồi train lại ít nhất hai seed. Chênh lệch giữa raw ban đầu và raw không rò rỉ mới ước lượng được tác động của phần rò rỉ trong thiết lập này.

## 4. Khác biệt theo lớp

| Lớp | AP50 raw trung bình | AP50 clean trung bình | Clean − raw |
|---|---:|---:|---:|
| car | 0.7375 | 0.7538 | +0.0163 |
| person | 0.8221 | 0.8275 | +0.0054 |
| truck | 0.6485 | 0.6242 | −0.0243 |
| bus | 0.6949 | 0.7149 | **+0.0200** |

Bus có mức tăng trung bình lớn nhất, nhưng chênh lệch giữa các seed cũng đáng kể; với hai seed chưa thể coi đây là kết luận chắc. `02_error_list_v1` có 17 issue `POSSIBLE_WRONG_CLASS`, phản ánh nguy cơ nhầm lớp xe; clean còn phụ thuộc vào cảnh báo model và ngân sách người duyệt. Số box cũng ít ở truck/bus (raw lần lượt 32/35, clean 41/31 theo thống kê token lớp bạn cung cấp), nên AP hai lớp này dễ dao động. Không thể khẳng định một nhóm lỗi cụ thể gây ra thay đổi AP nếu chưa đối chiếu từng nhãn sửa với AP theo lớp.

## 5. Clean còn lỗi và có phải dữ liệu sạch tối đa?

Hàng đợi có **121 ảnh**, vượt ngân sách duyệt 80 ảnh; mô phỏng đã duyệt 80 ảnh. Sau duyệt còn cảnh báo trên 34/193 ảnh (17.6%), nên pipeline kết luận **Hold**. Không còn lỗi bắt buộc ở danh sách cuối, nhưng cảnh báo ngữ nghĩa và box vẫn còn.

Vì vậy `clean` ở đây là dữ liệu đã qua tự động sửa và duyệt trong giới hạn ngân sách, **không phải nhãn chuẩn hoàn hảo hay chặn trên của dữ liệu sạch**. Muốn đo chặn trên, cần một biến thể oracle có người sửa toàn bộ ảnh và nhãn; kết quả clean hiện tại chỉ là một bước cải thiện có giới hạn. Cũng cần lưu ý pipeline đã xuất dataset Hold ở trạng thái nháp để phục vụ thí nghiệm train, không phải bản phát hành.
