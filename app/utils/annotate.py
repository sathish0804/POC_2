from typing import Dict, Any, List
import os
import cv2


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def draw_box(img, box, color=(0, 255, 0), label: str = None):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def annotate_and_save(frame_bgr, result: Dict[str, Any], tag: str, out_dir: str) -> None:
    activities = result.get("activities", [])
    if not activities:
        return
    out = frame_bgr.copy()
    for d in result.get("detections", []):
        draw_box(out, d.get("bbox", [0, 0, 0, 0]), (0, 255, 0) if d.get("name") == "person" else (255, 0, 0), f"{d.get('name')} {d.get('conf', 0.0):.2f}")
    for act in activities:
        # Skip drawing synthetic group person box to avoid confusion
        if str(act.get("person_id")) == "group":
            continue
        if act.get("object_bbox") is not None:
            draw_box(out, act.get("object_bbox"), (0, 165, 255), f"HOLDING {act.get('object')}")
        draw_box(out, act.get("person_bbox"), (0, 165, 255), f"person_{act.get('person_id')}")
    ensure_dir(out_dir)
    cv2.imwrite(os.path.join(out_dir, f"{tag}_activity.jpg"), out)
