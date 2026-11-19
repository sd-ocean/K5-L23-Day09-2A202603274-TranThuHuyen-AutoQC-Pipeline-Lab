"""Kiểm tra bổ sung: cảnh báo box có tỉ lệ khung hình cực đoan.

Tỉ lệ được tính theo pixel ảnh (không chỉ theo tọa độ chuẩn hóa), để ảnh
không vuông vẫn được đánh giá đúng. Đây là cảnh báo cho người xem, không tự sửa.
"""
from ..common import WARN, Issue, parse_yolo_line


MIN_ASPECT_RATIO = 0.1
MAX_ASPECT_RATIO = 10.0


def run(ds, cfg=None):
    issues = []
    for image_path in ds.image_files():
        stem = image_path.stem
        size = ds.image_size(stem)
        if not size:
            continue
        image_width, image_height = size
        if image_width <= 0 or image_height <= 0:
            continue
        for line_no, raw in ds.read_label_lines(stem) or []:
            try:
                cls, _, _, box_width, box_height = parse_yolo_line(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if not (0 <= cls < len(ds.classes)) or box_width <= 0 or box_height <= 0:
                continue
            ratio = (box_width * image_width) / (box_height * image_height)
            if ratio < MIN_ASPECT_RATIO or ratio > MAX_ASPECT_RATIO:
                issues.append(Issue(
                    "aspect_ratio", WARN, "BOX_EXTREME_ASPECT", stem,
                    f"tỉ lệ rộng/cao theo pixel {ratio:.3g} nằm ngoài khoảng "
                    f"[{MIN_ASPECT_RATIO:g}, {MAX_ASPECT_RATIO:g}]",
                    line_no, {"aspect_ratio": ratio},
                ))
    return issues
