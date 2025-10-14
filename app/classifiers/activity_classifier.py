from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from loguru import logger

from utils.geometry import compute_eye_aspect_ratio, iou, bbox_from_points, bbox_center, point_in_box


# Activity type IDs mapping (example):
# 1: Micro sleep
# 2: Sleeping
# 3: Using a cell phone
# 4: Writing while moving
# 5: Packing things before stop
# 7: Signal exchange with flag


class ActivityClassifier:
    """Rule-based activity classifier combining YOLO detections and MediaPipe landmarks.

    State is minimal and per-frame; for robust temporal logic, consider adding a tracker and buffers.
    """

    def __init__(self):
        # Thresholds
        self.ear_closed_threshold = 0.21
        self.micro_sleep_seconds = 2.0
        self.sleep_seconds = 10.0
        self.iou_hand_phone_threshold = 0.1

        # Simple per-run temporal counters (seconds of sustained condition)
        self.eyes_closed_accum = 0.0
        self.head_stationary_accum = 0.0
        self.hand_bag_overlap_accum = 0.0
        self.writing_motion_accum = 0.0
        self.flag_exchange_accum = 0.0

        # Naive last-state for motion estimation
        self._last_hand_centers: Optional[List[tuple]] = None
        self._last_head_center: Optional[tuple] = None
        self._last_ts: Optional[float] = None
        logger.debug("[ActivityClassifier] initialized")

    def classify_frame(
        self,
        frame_bgr: np.ndarray,
        mp_out: Dict[str, Any],
        detections: List[Tuple[int, float, tuple]],
        timestamp: float
    ) -> List[Tuple[int, str, Optional[int]]]:
        """Return list of detected activities for the frame.

        Each item: (activityType, description, peopleCount)
        """
        # Dynamic sec_step from timestamps (fallback to 1.0 if unavailable)
        if self._last_ts is None or timestamp <= self._last_ts:
            sec_step = 1.0
        else:
            sec_step = min(5.0, max(0.05, timestamp - self._last_ts))
        self._last_ts = timestamp

        activities: List[Tuple[int, str, Optional[int]]] = []

        eyes_closed, head_center = self._estimate_eye_closure_and_head(mp_out)
        # Head tilt forward heuristic: downward movement of head center y
        head_tilt_forward = False
        if head_center is not None and self._last_head_center is not None:
            dy = head_center[1] - self._last_head_center[1]
            head_tilt_forward = dy > 0.01

        if eyes_closed:
            self.eyes_closed_accum += sec_step
        else:
            self.eyes_closed_accum = 0.0

        head_stationary = self._is_head_stationary(head_center)
        if head_stationary:
            self.head_stationary_accum += sec_step
        else:
            self.head_stationary_accum = 0.0

        # 1 & 2: Micro sleep vs Sleeping
        if self.eyes_closed_accum >= self.sleep_seconds and self.head_stationary_accum >= self.sleep_seconds:
            activities.append((2, "Sleeping detected: eyes closed >10s and head stationary.", 1))
        elif self.eyes_closed_accum >= self.micro_sleep_seconds and head_tilt_forward:
            activities.append((1, "Micro sleep detected: eyes closed >2s and head tilting forward.", 1))

        # Extract hands bboxes from landmarks for IoU checks
        hand_boxes = self._get_hand_bboxes(mp_out)

        # Using cell phone: IoU(hand, phone)
        phone_boxes = [box for (cid, _, box) in detections if self._is_class_phone(cid)]
        if hand_boxes and phone_boxes:
            for hb in hand_boxes:
                for pb in phone_boxes:
                    if iou(hb, pb) > self.iou_hand_phone_threshold:
                        activities.append((3, "Using cell phone: hand overlaps phone bbox.", 1))
                        break

        # Writing while moving: hand motion around pen/notebook object
        pen_like_boxes = [box for (cid, _, box) in detections if self._is_class_pen_or_notebook(cid)]
        writing_motion = self._hand_motion_near_targets(hand_boxes, pen_like_boxes)
        if writing_motion:
            self.writing_motion_accum += sec_step
        else:
            self.writing_motion_accum = 0.0
        if self.writing_motion_accum >= 2.0:
            activities.append((4, "Writing while moving: repetitive hand motion near pen/notebook.", 1))

        # Packing: frequent overlap of hand and bag/backpack
        bag_boxes = [box for (cid, _, box) in detections if self._is_class_bag(cid)]
        if hand_boxes and bag_boxes:
            overlap = any(iou(hb, bb) > 0.05 for hb in hand_boxes for bb in bag_boxes)
            if overlap:
                self.hand_bag_overlap_accum += sec_step
            else:
                self.hand_bag_overlap_accum = 0.0
            if self.hand_bag_overlap_accum >= 3.0:
                activities.append((5, "Packing detected: frequent hand-bag overlaps.", 1))

        # Calling signals logic removed

        # Signal exchange with flag: presence of flag + hand interacting
        flag_boxes = [box for (cid, _, box) in detections if self._is_class_flag(cid)]
        if hand_boxes and flag_boxes:
            interaction = any(iou(hb, fb) > 0.05 for hb in hand_boxes for fb in flag_boxes)
            if interaction:
                self.flag_exchange_accum += sec_step
            else:
                self.flag_exchange_accum = 0.0
            if self.flag_exchange_accum >= 1.0:
                activities.append((7, "Signal exchange with flag: flag present and hand interacts.", 1))

        return activities


# Module import log
logger.debug(f"[{__name__}] module loaded")

    # ----- Helpers -----
    def _estimate_eye_closure_and_head(self, mp_out: Dict[str, Any]) -> tuple:
        face_res = mp_out.get('face')
        if not face_res or not face_res.multi_face_landmarks:
            return False, None
        # Approximate eye landmarks indices for FaceMesh (iris refined)
        # Using a simple subset: left eye [33, 160, 158, 133, 153, 144], right eye [362, 385, 387, 263, 373, 380]
        face_landmarks = face_res.multi_face_landmarks[0]
        pts = [(lm.x, lm.y) for lm in face_landmarks.landmark]

        def get_eye(indices):
            return [(pts[i][0], pts[i][1]) for i in indices]

        left_idx = [33, 160, 158, 133, 153, 144]
        right_idx = [362, 385, 387, 263, 373, 380]
        if max(left_idx + right_idx) >= len(pts):
            return False, None
        left_eye = get_eye(left_idx)
        right_eye = get_eye(right_idx)
        left_ear = compute_eye_aspect_ratio(left_eye)
        right_ear = compute_eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        eyes_closed = ear < self.ear_closed_threshold

        # Head center: between eyes
        head_center = ((left_eye[0][0] + right_eye[3][0]) / 2.0, (left_eye[0][1] + right_eye[3][1]) / 2.0)
        return eyes_closed, head_center

    def _is_head_stationary(self, head_center: Optional[tuple]) -> bool:
        if head_center is None:
            return False
        stationary = False
        if self._last_head_center is not None:
            dx = abs(head_center[0] - self._last_head_center[0])
            dy = abs(head_center[1] - self._last_head_center[1])
            stationary = (dx + dy) < 0.01
        self._last_head_center = head_center
        return stationary

    def _get_hand_bboxes(self, mp_out: Dict[str, Any]) -> List[tuple]:
        hands_res = mp_out.get('hands')
        if not hands_res or not hands_res.multi_hand_landmarks:
            return []
        boxes = []
        for hand_lm in hands_res.multi_hand_landmarks:
            pts = [(lm.x, lm.y) for lm in hand_lm.landmark]
            boxes.append(bbox_from_points(pts))
        return boxes

    def _hand_motion_near_targets(self, hand_boxes: List[tuple], targets: List[tuple]) -> bool:
        if not hand_boxes or not targets:
            self._last_hand_centers = None
            return False
        centers = [bbox_center(b) for b in hand_boxes]
        moving = False
        if self._last_hand_centers is not None:
            for c_prev, c_now in zip(self._last_hand_centers, centers):
                dx = abs(c_now[0] - c_prev[0])
                dy = abs(c_now[1] - c_prev[1])
                near_target = any(point_in_box(c_now, t) for t in targets)
                if near_target and (dx + dy) > 0.02:
                    moving = True
                    break
        self._last_hand_centers = centers
        return moving

    # ----- Class ID helpers (placeholder; customize per model classes) -----
    def _is_class_phone(self, class_id: int) -> bool:
        # COCO class id for cell phone is 67 in some mappings; YOLOv8 uses names dict. Customize as needed.
        return class_id in {67}

    def _is_class_pen_or_notebook(self, class_id: int) -> bool:
        # Requires custom-trained model; placeholder ids
        return class_id in {900, 901}

    def _is_class_bag(self, class_id: int) -> bool:
        # backpack class is 24 in COCO; bag/handbag may be 26. Adjust per model.
        return class_id in {24, 26}

    def _is_class_flag(self, class_id: int) -> bool:
        # Requires custom flag class id; placeholder
        return class_id in {902}

    # Calling signals helper removed