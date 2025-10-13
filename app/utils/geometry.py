from typing import List, Tuple
import math


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


def euclidean_distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_eye_aspect_ratio(eye_points: List[Point]) -> float:
    """Compute EAR given 6 eye landmark points in order [p1,p2,p3,p4,p5,p6].

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    if len(eye_points) != 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye_points
    numerator = euclidean_distance(p2, p6) + euclidean_distance(p3, p5)
    denominator = 2.0 * euclidean_distance(p1, p4)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def iou(box_a: BBox, box_b: BBox) -> float:
    """Intersection over Union for 2 bboxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def angle_between_vectors(v1: Point, v2: Point) -> float:
    """Return angle in degrees between vectors v1 and v2."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos))


def angle_at_joint(p1: Point, p2: Point, p3: Point) -> float:
    """Angle at p2 made by (p1->p2) and (p3->p2)."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    return angle_between_vectors(v1, v2)


def bbox_from_points(points: List[Point]) -> BBox:
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_center(box: BBox) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_box(pt: Point, box: BBox) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)
