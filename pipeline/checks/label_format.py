"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    rules = cfg["label_format"]
    issues = []
    for path in ds.label_files():
        stem = path.stem
        size = ds.image_size(stem)
        seen = []
        for line_no, raw in ds.read_label_lines(stem) or []:
            try:
                cls, cx, cy, w, h = parse_yolo_line(raw)
            except (TypeError, ValueError) as exc:
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem,
                                    f"dòng nhãn không hợp lệ: {exc}", line_no))
                continue
            if cls < 0 or cls >= len(ds.classes):
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem,
                                    f"id lớp {cls} ngoài phạm vi 0..{len(ds.classes)-1}", line_no))
                continue
            if w <= 0 or h <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem,
                                    "chiều rộng hoặc chiều cao box không dương", line_no))
                continue

            box = (cx, cy, w, h)
            left, top = cx - w / 2, cy - h / 2
            right, bottom = cx + w / 2, cy + h / 2
            overflow = max(0.0, -left, -top, right - 1.0, bottom - 1.0)
            if overflow > rules["overflow_tol"]:
                issues.append(Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", stem,
                                    f"box tràn khỏi ảnh {overflow:.6g}", line_no,
                                    {"overflow": overflow}))

            if any(old_cls == cls and iou_xywh(box, old_box) > rules["dup_box_iou"]
                   for old_cls, old_box in seen):
                issues.append(Issue("label_format", ERROR, "DUP_BOX", stem,
                                    "box gần trùng với box cùng lớp ở dòng trước", line_no))
            seen.append((cls, box))

            if size is not None:
                width_px, height_px = size
                if min(w * width_px, h * height_px) < rules["min_box_px"]:
                    issues.append(Issue("label_format", WARN, "BOX_TINY", stem,
                                        "cạnh ngắn của box nhỏ hơn ngưỡng pixel", line_no))
            if w * h > rules["max_box_frac"]:
                issues.append(Issue("label_format", WARN, "BOX_HUGE", stem,
                                    "box phủ phần lớn ảnh", line_no))
    return issues
