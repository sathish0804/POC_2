from typing import Dict, Tuple, List, Optional
import math

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]


def rect_center(rect: Rect) -> Point:
    x1, y1, x2, y2 = rect
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_to_rect_min_distance(pt: Point, rect: Rect) -> float:
    x, y = pt
    x1, y1, x2, y2 = rect
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return math.hypot(dx, dy)


def l2(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_face_zones(face_bbox: Rect) -> Dict[str, Rect]:
    fx1, fy1, fx2, fy2 = face_bbox
    fw = max(1.0, fx2 - fx1)
    fh = max(1.0, fy2 - fy1)

    ear_band_y1 = fy1 + 0.2 * fh
    ear_band_y2 = fy1 + 0.75 * fh
    ear_band_w = 0.18 * fw

    left_ear_zone = (fx1 - 0.10 * fw, ear_band_y1, fx1 + ear_band_w, ear_band_y2)
    right_ear_zone = (fx2 - ear_band_w, ear_band_y1, fx2 + 0.10 * fw, ear_band_y2)

    mouth_x1 = fx1 + 0.20 * fw
    mouth_x2 = fx1 + 0.80 * fw
    mouth_y1 = fy1 + 0.50 * fh
    mouth_y2 = fy1 + 0.95 * fh
    mouth_zone = (mouth_x1, mouth_y1, mouth_x2, mouth_y2)

    return {
        "left_ear_zone": left_ear_zone,
        "right_ear_zone": right_ear_zone,
        "mouth_zone": mouth_zone,
    }


def select_hand_points(hand_landmarks: Dict[str, Dict[str, float]]) -> List[Point]:
    key_indices = ["0", "4", "8", "20"]
    pts: List[Point] = []
    for k in key_indices:
        if k in hand_landmarks:
            pts.append((float(hand_landmarks[k]["x"]), float(hand_landmarks[k]["y"])))
    return pts


def infer_phone_usage_from_landmarks(
    person_box: Tuple[float, float, float, float],
    face_bbox: Optional[Tuple[float, float, float, float]],
    hands_list: List[Dict[str, Dict[str, float]]],
) -> Tuple[bool, Dict[str, object]]:
    if not hands_list or face_bbox is None:
        return False, {"reason": "missing_face_or_hands"}

    fx1, fy1, fx2, fy2 = map(float, face_bbox)
    zones = compute_face_zones((fx1, fy1, fx2, fy2))

    px1, py1, px2, py2 = map(float, person_box)
    diag = math.hypot(px2 - px1, py2 - py1)
    ear_close_thresh = 0.06 * diag
    mouth_close_thresh = 0.09 * diag

    for hand in hands_list:
        pts = select_hand_points(hand)
        for zone_name, rect in zones.items():
            for pt in pts:
                d = point_to_rect_min_distance(pt, rect)
                if d <= 0.0:
                    # Always classify as cell phone; walkie-talkie logic removed
                    device_type = "cell phone"
                    return True, {
                        "rule": f"hand_in_{zone_name}",
                        "distance_px": 0.0,
                        "zone": zone_name,
                        "device_type": device_type,
                    }
            cx, cy = rect_center(rect)
            for pt in pts:
                d = l2(pt, (cx, cy))
                if ("ear" in zone_name and d < ear_close_thresh):
                    return True, {
                        "rule": f"hand_near_{zone_name}",
                        "distance_px": float(d),
                        "zone": zone_name,
                        "device_type": "cell phone",
                    }
                elif ("mouth" in zone_name and d < mouth_close_thresh):
                    # Reintroduce walkie-talkie mapping for mouth-zone proximity
                    return True, {
                        "rule": f"hand_near_{zone_name}",
                        "distance_px": float(d),
                        "zone": zone_name,
                        "device_type": "walkie_talkie",
                    }
    return False, {"reason": "no_hand_near_face_zones"}
