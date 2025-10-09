from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from utils.geometry import iou as iou_xyxy


class SimpleTracker:
    def __init__(self, iou_match_thresh: float = 0.30, ttl_s: float = 30.0):
        self.iou_match_thresh = float(iou_match_thresh)
        self.ttl_s = float(ttl_s)
        self._next_id = 1
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def _match_track(self, box: np.ndarray, ts: float, used_track_ids: set) -> Optional[int]:
        best_id = None
        best_iou = 0.0
        for tid, t in self._tracks.items():
            if tid in used_track_ids:
                continue
            iou_val = iou_xyxy(tuple(t["bbox"]), tuple(box.tolist()))
            if iou_val > best_iou:
                best_iou = iou_val
                best_id = tid
        if best_id is not None and best_iou >= self.iou_match_thresh:
            return best_id
        return None

    def assign(self, person_boxes: List[np.ndarray], ts: float) -> List[int]:
        assigned: List[int] = []
        used: set = set()
        for box in person_boxes:
            tid = self._match_track(box, ts, used)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
            self._tracks[tid] = {
                "bbox": np.array(box, dtype=float).tolist(),
                "last_ts": float(ts),
            }
            used.add(tid)
            assigned.append(tid)
        self.cleanup(ts)
        return assigned

    def cleanup(self, ts: float) -> None:
        to_del = []
        for tid, t in self._tracks.items():
            if (ts - float(t.get("last_ts", ts))) > self.ttl_s:
                to_del.append(tid)
        for tid in to_del:
            self._tracks.pop(tid, None)


class SleepTracker:
    def __init__(
        self,
        eye_thresh: float = 0.18,
        head_down_deg_thresh: float = 100.0,
        micro_max_s: float = 15.0 * 60.0,
        min_duration_s: float = 10.0,
    ):
        self.eye_thresh = float(eye_thresh)
        self.head_down_deg_thresh = float(head_down_deg_thresh)
        self.micro_max_s = float(micro_max_s)
        self.min_duration_s = float(min_duration_s)
        self._state: Dict[int, Dict[str, Any]] = {}

    def _should_sleep(self, eye_openness: Optional[float], head_down_angle_deg: Optional[float], engaged: bool) -> bool:
        if engaged:
            return False
        if eye_openness is None or head_down_angle_deg is None:
            return False
        eyes_closed = eye_openness <= self.eye_thresh
        head_down = float(head_down_angle_deg) >= self.head_down_deg_thresh
        return bool(eyes_closed and head_down)

    def update(
        self,
        track_id: int,
        ts: float,
        person_bbox: List[float],
        eye_openness: Optional[float],
        head_down_angle_deg: Optional[float],
        engaged: bool,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        st = self._state.get(track_id, {"active": False, "start_ts": None})
        is_sleeping_now = self._should_sleep(eye_openness, head_down_angle_deg, engaged)

        if is_sleeping_now and not st.get("active", False):
            st = {
                "active": True,
                "start_ts": float(ts),
                "last_ts": float(ts),
                "start_bbox": list(person_bbox),
            }
        elif is_sleeping_now and st.get("active", False):
            st["last_ts"] = float(ts)
        elif (not is_sleeping_now) and st.get("active", False):
            start_ts = float(st.get("start_ts", ts))
            end_ts = float(st.get("last_ts", ts))
            duration = max(0.0, end_ts - start_ts)
            if duration >= self.min_duration_s:
                activity_name = "micro_sleep" if duration < self.micro_max_s else "sleep"
                events.append({
                    "activity": activity_name,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "duration_s": duration,
                    "track_id": int(track_id),
                    "person_bbox": list(person_bbox),
                    "evidence_rule": "sleep_eye_closed_head_down",
                })
            st = {"active": False, "start_ts": None}

        self._state[track_id] = st
        return events

    def finalize(self, ts: Optional[float] = None) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for track_id, st in list(self._state.items()):
            if st.get("active", False):
                start_ts = float(st.get("start_ts", 0.0))
                end_ts = float(ts if ts is not None else st.get("last_ts", start_ts))
                duration = max(0.0, end_ts - start_ts)
                if duration >= self.min_duration_s:
                    activity_name = "micro_sleep" if duration < self.micro_max_s else "sleep"
                    events.append({
                        "activity": activity_name,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "duration_s": duration,
                        "track_id": int(track_id),
                        "person_bbox": list(st.get("start_bbox", [])),
                        "evidence_rule": "sleep_eye_closed_head_down",
                    })
            self._state[track_id] = {"active": False, "start_ts": None}
        return events


class ActivityHeuristicTracker:
    """Lightweight, per-person heuristic tracker for activities 4 and 5.

    - Writing while moving (4): repetitive small-range hand motion.
    - Packing (5): sustained hand overlap with bag/backpack.

    This is intentionally simple and CPU-friendly. Thresholds assume ~1 FPS.
    """

    def __init__(
        self,
        writing_window_s: float = 2.0,
        writing_min_path_px: float = 40.0,
        writing_max_radius_px: float = 35.0,
        packing_window_s: float = 2.0,
        iou_overlap_thresh: float = 0.05,
    ):
        self.writing_window_s = float(writing_window_s)
        self.writing_min_path_px = float(writing_min_path_px)
        self.writing_max_radius_px = float(writing_max_radius_px)
        self.packing_window_s = float(packing_window_s)
        self.iou_overlap_thresh = float(iou_overlap_thresh)

        # Per track state
        self._state: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _center_of_box(box: List[float]) -> tuple:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _path_length(points: List[tuple]) -> float:
        total = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            total += (dx * dx + dy * dy) ** 0.5
        return total

    @staticmethod
    def _bbox_of_points(points: List[tuple]) -> List[float]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return [min(xs), min(ys), max(xs), max(ys)]

    def _ensure_track(self, track_id: int) -> Dict[str, Any]:
        st = self._state.get(track_id)
        if st is None:
            st = {
                "last_ts": None,
                "hand_centers": [],  # list of (x,y) within window
                "writing_accum": 0.0,
                "packing_accum": 0.0,
            }
            self._state[track_id] = st
        return st

    def update(
        self,
        *,
        track_id: int,
        ts: float,
        person_bbox: List[float],
        hand_boxes_frame: List[List[float]],
        bag_boxes_frame: List[List[float]],
        pose_points: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        st = self._ensure_track(track_id)
        st["last_ts"] = float(ts)

        # Aggregate hand centers for writing/gesture analysis
        for hb in hand_boxes_frame:
            st["hand_centers"].append(self._center_of_box(hb))
        # Keep only recent points covering the max window
        if len(st["hand_centers"]) > 120:
            st["hand_centers"] = st["hand_centers"][-120:]

        # Writing: repetitive small-range motion
        if len(st["hand_centers"]) >= 3:
            path = self._path_length(st["hand_centers"])  # pixels across whole buffer
            bx1, by1, bx2, by2 = self._bbox_of_points(st["hand_centers"])
            radius = max(bx2 - bx1, by2 - by1)
            if path >= self.writing_min_path_px and radius <= self.writing_max_radius_px:
                st["writing_accum"] += 1.0  # ~1 sec per sampled frame
            else:
                st["writing_accum"] = 0.0
        if st["writing_accum"] >= self.writing_window_s:
            events.append({
                "activity": "writing",
                "evidence_rule": "hand_small_range_repetitive_motion",
            })
            st["writing_accum"] = 0.0

        # Packing: sustained hand-bag overlaps
        if hand_boxes_frame and bag_boxes_frame:
            overlap = any(
                iou_xyxy(tuple(hb), tuple(bb)) > self.iou_overlap_thresh
                for hb in hand_boxes_frame for bb in bag_boxes_frame
            )
            if overlap:
                st["packing_accum"] += 1.0
            else:
                st["packing_accum"] = 0.0
        if st["packing_accum"] >= self.packing_window_s:
            events.append({
                "activity": "packing",
                "evidence_rule": "hand_bag_overlap_sustained",
            })
            st["packing_accum"] = 0.0

        # Calling signals logic removed

        return events
