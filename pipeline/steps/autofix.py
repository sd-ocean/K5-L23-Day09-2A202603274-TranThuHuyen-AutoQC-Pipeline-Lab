"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line


def fix(ds, issues, cfg):
    changes = []
    errors = [i for i in issues if i.severity == ERROR]
    removed = set()
    remove_codes = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    by_image = defaultdict(lambda: defaultdict(set))

    for issue in errors:
        if issue.code in remove_codes:
            stem = issue.image
            if stem in removed:
                continue
            image = ds.image_path(stem)
            label = ds.label_path(stem)
            before = str(image.name) if image else stem
            if image:
                image.unlink(missing_ok=True)
            label.unlink(missing_ok=True)
            removed.add(stem)
            changes.append(dict(image=stem, line="", action="remove_image", before=before,
                                after="đã xóa ảnh và nhãn khỏi dataset làm việc"))
        elif issue.code in drop_codes | {"BOX_OUT_OF_IMAGE"} and issue.line is not None:
            by_image[issue.image][issue.line].add(issue.code)
        elif issue.code == "MISSING_LABEL_FILE":
            changes.append(dict(image=issue.image, line="", action="needs_label",
                                before="(không có file)", after="chuyển người gán"))
        elif issue.code == "BAD_CLASS_NAME":
            # Chỉ ghi YAML một lần dù check có thể phát nhiều issue cấp dataset.
            if not any(row["action"] == "rename_classes" for row in changes):
                old = ds.yaml_path.read_text(encoding="utf-8", errors="replace") if ds.yaml_path.exists() else "(không có data.yaml)"
                doc = yaml.safe_load(old) if old.startswith(("{", "path:", "names:")) else None
                if not isinstance(doc, dict):
                    doc = {}
                doc["names"] = {i: name for i, name in enumerate(cfg["classes"])}
                ds.yaml_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
                changes.append(dict(image="data.yaml", line="", action="rename_classes", before=old,
                                    after=str(cfg["classes"])))

    for stem in sorted(by_image):
        if stem in removed:
            continue
        label = ds.label_path(stem)
        if not label.exists():
            continue
        original = label.read_text(encoding="utf-8", errors="replace").splitlines()
        line_issues = by_image[stem]
        kept = []
        touched = {}
        for number, raw in enumerate(original, 1):
            codes = line_issues.get(number, set())
            if not codes:
                kept.append(raw)
                continue
            if codes & drop_codes:
                action = "drop_line"
                after = ""
            else:
                try:
                    cls, cx, cy, w, h = parse_yolo_line(raw)
                except (TypeError, ValueError):
                    action, after = "drop_line", ""
                else:
                    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
                    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
                    nw, nh = x2 - x1, y2 - y1
                    if nw <= 0 or nh <= 0:
                        action, after = "drop_line", ""
                    else:
                        ncx, ncy = (x1 + x2) / 2, (y1 + y2) / 2
                        action = "clip_box"
                        after = f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"
            touched[number] = (action, raw, after)
            if action != "drop_line":
                kept.append(after)
        if touched:
            label.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            for number in sorted(touched):
                action, before, after = touched[number]
                changes.append(dict(image=stem, line=number, action=action, before=before, after=after))
    return changes
