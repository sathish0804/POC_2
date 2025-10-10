from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import os
import cv2
import numpy as np

from models.activity_event import ActivityEvent
from services.yolo_service import YoloService
from services.mediapipe_service import MediaPipeService
from utils.ocr_utils import OcrUtils
from utils.video_utils import sample_video_frames, get_video_duration_str, get_video_filename, get_expected_sampled_frames
from utils.geometry import iou
from utils.phone_logic import infer_phone_usage_from_landmarks
from utils.annotate import annotate_and_save
from utils.trackers import SimpleTracker, SleepTracker, ActivityHeuristicTracker
from utils.eye_metrics import compute_ear, eye_open_probability
from utils.sleep_decision import SleepDecisionMachine, SleepDecisionConfig
from utils.debug_overlay import save_debug_overlay
from utils.flag_utils import detect_green_flags
from utils.window_utils import detect_window_regions
from loguru import logger


@dataclass
class ActivityPipeline:
    trip_id: str
    crew_name: str
    crew_id: str
    crew_role: int
    yolo_weights: str
    sample_fps: int = 1
    enable_ocr: bool = True
    verbose: bool = False
    max_frames: int = 0
    # Sleep thresholds
    sleep_eye_thresh: float = 0.18
    sleep_headdown_deg: float = 100.0
    sleep_micro_max_min: float = 15.0
    sleep_min_duration: float = 10.0
    # Person post-processing thresholds (to reduce false extra persons)
    person_min_conf: float = 0.45
    person_min_area_frac: float = 0.015
    person_nms_iou: float = 0.70
    # Phone robustness thresholds
    phone_max_area_person_frac: float = 0.15
    phone_ar_min: float = 0.30
    phone_ar_max: float = 3.00
    phone_min_conf: float = 0.50
    # Glare/torch suppression for phone
    phone_glare_v_thresh: int = 240
    phone_glare_s_thresh: int = 60
    phone_glare_frac_max: float = 0.35
    phone_edge_density_min: float = 0.02
    # Advanced sleep logic
    use_advanced_sleep: bool = False
    save_debug_overlays: bool = False
    # Phone inference robustness
    phone_hand_iou_min_frac: float = 0.15
    phone_infer_min_face_frac: float = 0.02
    phone_infer_suppress_head_down: bool = True
    phone_infer_head_down_deg: float = 35.0
    phone_infer_max_hand_y_frac: float = 0.65  # suppress if all hand points below this fraction of person height
    # Packing detection config (scalable knobs)
    pack_iou_overlap_thresh: float = 0.20   # 0.15–0.25
    pack_window_s: float = 3.0              # seconds of sustained overlap
    bag_min_score: float = 0.50
    bag_area_frac_min: float = 0.01         # 1% of person area
    bag_area_frac_max: float = 0.20         # 20% of person area
    bag_torso_band_frac: float = 0.60       # torso band height fraction of person
    # Writing detection config (scalable)
    write_window_s: float = 3.0
    write_min_path_px: float = 100.0
    write_max_radius_px: float = 25.0
    write_lap_band_frac: float = 0.40       # bottom 40% of person bbox

    def process_video(self, video_path: str, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[ActivityEvent]:
        if self.verbose:
            logger.debug("[Pipeline] Initializing services (YOLO, MediaPipe, OCR)...")
        # Allow adjusting YOLO confidence via env var for quick sweeps (optional)
        try:
            import os as _os
            _yolo_conf = float(_os.getenv("YOLO_CONF", "0.25"))
            _yolo_iou = float(_os.getenv("YOLO_IOU", "0.45"))
        except Exception:
            _yolo_conf, _yolo_iou = 0.25, 0.45
        yolo = YoloService(self.yolo_weights, conf=_yolo_conf, iou=_yolo_iou)
        mp_service = MediaPipeService()
        ocr = OcrUtils() if self.enable_ocr else None
        # Antenna-based relabeling removed

        events: List[ActivityEvent] = []
        file_duration = get_video_duration_str(video_path)
        filename = get_video_filename(video_path)

        tracker = SimpleTracker()
        sleep_tracker = SleepTracker(
            eye_thresh=self.sleep_eye_thresh,
            head_down_deg_thresh=self.sleep_headdown_deg,
            micro_max_s=float(self.sleep_micro_max_min) * 60.0,
            min_duration_s=self.sleep_min_duration,
        )
        act_tracker = ActivityHeuristicTracker(
            writing_window_s=self.write_window_s,
            writing_min_path_px=self.write_min_path_px,
            writing_max_radius_px=self.write_max_radius_px,
            packing_window_s=self.pack_window_s,
            iou_overlap_thresh=self.pack_iou_overlap_thresh,
        )
        sleep_decider = SleepDecisionMachine(SleepDecisionConfig()) if self.use_advanced_sleep else None

        processed = 0
        # Estimate total sampled frames for progress reporting
        try:
            expected_total = int(get_expected_sampled_frames(video_path, self.sample_fps))
            if self.max_frames and self.max_frames > 0:
                expected_total = min(expected_total, int(self.max_frames))
        except Exception:
            expected_total = 0
        # initial callback
        if callable(progress_cb):
            try:
                progress_cb({"processed": processed, "total": expected_total})
            except Exception:
                pass
        for index, ts, frame_bgr in sample_video_frames(video_path, self.sample_fps):
            if self.max_frames and processed >= self.max_frames:
                if self.verbose:
                    logger.debug(f"[Pipeline] Reached max_frames={self.max_frames}, stopping early.")
                break
            if self.verbose:
                logger.debug(f"[Frame {index}] ts={ts:.2f}s: detection + landmarks + per-person crops")

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = yolo.detect(frame_bgr)
            mp_out = mp_service.process(frame_rgb)
            # Detect green flag candidates and window regions once per frame
            green_flags = detect_green_flags(frame_bgr)
            window_regions = detect_window_regions(frame_bgr)
            if ocr is not None:
                date_str, time_str = ocr.extract_date_time(frame_bgr)
            else:
                date_str, time_str = "", ""

            # Persons and objects (apply stricter filtering + NMS on persons)
            H, W = frame_bgr.shape[0], frame_bgr.shape[1]
            frame_area = float(H * W)
            min_area = max(1.0, self.person_min_area_frac * frame_area)

            person_candidates = [
                (cid, score, box)
                for (cid, score, box) in detections
                if cid == 0 and float(score) >= float(self.person_min_conf)
                and max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1])) >= min_area
            ]

            # Greedy NMS to deduplicate highly-overlapping person boxes
            person_candidates.sort(key=lambda d: float(d[1]), reverse=True)
            persons: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
            for det in person_candidates:
                _, _, b = det
                if all(iou(b, kept[2]) < self.person_nms_iou for kept in persons):
                    persons.append(det)
            # Optional validation with MediaPipe landmarks to drop false person boxes
            face_res = None
            hands_res = None
            pose_res = None
            try:
                face_res = mp_out.get('face')
                hands_res = mp_out.get('hands')
                pose_res = mp_out.get('pose')
            except Exception:
                face_res = hands_res = pose_res = None

            def _collect_points() -> Tuple[list, list, list]:
                pts_face, pts_hands, pts_pose = [], [], []
                try:
                    if face_res and getattr(face_res, 'multi_face_landmarks', None):
                        for fl in face_res.multi_face_landmarks:
                            for lm in fl.landmark:
                                pts_face.append((lm.x * W, lm.y * H))
                except Exception:
                    pass
                try:
                    if hands_res and getattr(hands_res, 'multi_hand_landmarks', None):
                        for hl in hands_res.multi_hand_landmarks:
                            for lm in hl.landmark:
                                pts_hands.append((lm.x * W, lm.y * H))
                except Exception:
                    pass
                try:
                    if pose_res and getattr(pose_res, 'pose_landmarks', None):
                        for lm in pose_res.pose_landmarks.landmark:
                            pts_pose.append((lm.x * W, lm.y * H))
                except Exception:
                    pass
                return pts_face, pts_hands, pts_pose

            pts_face, pts_hands, pts_pose = _collect_points()
            have_any_landmarks = bool(pts_face or pts_hands or pts_pose)

            if have_any_landmarks and persons:
                def _count_inside(box, pts) -> int:
                    x1, y1, x2, y2 = box
                    cnt = 0
                    for (px, py) in pts:
                        if x1 <= px <= x2 and y1 <= py <= y2:
                            cnt += 1
                    return cnt

                validated: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
                for det in persons:
                    _, _, b = det
                    c_face = _count_inside(b, pts_face)
                    c_hands = _count_inside(b, pts_hands)
                    c_pose = _count_inside(b, pts_pose)
                    if (c_hands >= 8) or (c_pose >= 10) or (c_face >= 20):
                        validated.append(det)
                # Keep at least one person if all filtered out (avoid over-pruning)
                if validated:
                    persons = validated
            # Keep other detected classes as-is
            objects = [(cid, score, box) for (cid, score, box) in detections if cid != 0]

            # Assign per-person tracking ids by IoU over person boxes
            person_boxes_np = [np.array(b, dtype=float) for (_, _, b) in persons]
            track_ids = tracker.assign(person_boxes_np, float(ts)) if person_boxes_np else []

            # Build per-person analysis, annotate and collect activities
            per_frame_activities: List[Dict[str, Any]] = []

            for pid, (track_id, (_, _, pb)) in enumerate(zip(track_ids, persons), start=1):
                x1, y1, x2, y2 = map(int, pb)
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame_bgr.shape[1] - 1, x2); y2 = min(frame_bgr.shape[0] - 1, y2)
                crop = frame_bgr[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                # Extract landmarks in crop: derive hands and face boxes/points
                hands_res = mp_out.get('hands')
                face_res = mp_out.get('face')
                pose_res = mp_out.get('pose')

                # Build hand landmarks relative to crop if inside crop
                hands_list = []
                if hands_res and hands_res.multi_hand_landmarks:
                    for hand_lm in hands_res.multi_hand_landmarks:
                        pts = []
                        for lm in hand_lm.landmark:
                            px = lm.x * frame_bgr.shape[1]
                            py = lm.y * frame_bgr.shape[0]
                            if x1 <= px <= x2 and y1 <= py <= y2:
                                pts.append({"x": px - x1, "y": py - y1, "z": lm.z})
                        if pts:
                            hands_list.append({str(i): p for i, p in enumerate(pts)})

                # Face bbox in crop
                face_bbox_crop = None
                if face_res and face_res.multi_face_landmarks:
                    # Use first face only; compute bbox
                    xs, ys = [], []
                    for lm in face_res.multi_face_landmarks[0].landmark:
                        px = lm.x * frame_bgr.shape[1]
                        py = lm.y * frame_bgr.shape[0]
                        if x1 <= px <= x2 and y1 <= py <= y2:
                            xs.append(px - x1)
                            ys.append(py - y1)
                    if xs and ys:
                        fx1, fy1, fx2, fy2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
                        face_bbox_crop = (fx1, fy1, fx2, fy2)

                # Hand-object IoU (full-frame hands -> simplified via crop hands bbox)
                # Derive crop hand bbox
                hand_boxes_crop = []
                for hand in hands_list:
                    xs = [p["x"] for p in hand.values()]
                    ys = [p["y"] for p in hand.values()]
                    if xs and ys:
                        hx1, hy1, hx2, hy2 = min(xs), min(ys), max(xs), max(ys)
                        pad = 10.0
                        hand_boxes_crop.append((hx1 - pad, hy1 - pad, hx2 + pad, hy2 + pad))

                # Transform object boxes into crop space when overlapping person
                phone_near_person = False
                phone_bbox_crop = None
                held_object = None
                held_object_bbox = None
                # Flag variables
                flag_interaction_emitted = False

                for (cid, score, ob) in objects:
                    # Person overlap check
                    if iou(pb, ob) <= 0.0:
                        continue
                    name = "phone" if cid == 67 else "other"
                    # If it's a phone, check proximity and hand overlap
                    if cid == 67:  # phone
                        if float(score) < float(self.phone_min_conf):
                            continue
                        # If phone box overlaps crop and passes size/shape sanity checks
                        ox1, oy1, ox2, oy2 = ob
                        pw = max(1.0, float(x2 - x1)); ph = max(1.0, float(y2 - y1))
                        person_area = pw * ph
                        ow = max(1.0, float(ox2 - ox1)); oh = max(1.0, float(oy2 - oy1))
                        phone_area = ow * oh
                        area_frac = phone_area / max(1.0, person_area)
                        ar = ow / oh
                        # Skip huge or very odd aspect "phones" (e.g., monitors)
                        if area_frac > self.phone_max_area_person_frac or ar < self.phone_ar_min or ar > self.phone_ar_max:
                            continue
                        # Glare/torch suppression check on phone patch
                        try:
                            gx1 = max(0, int(ox1)); gy1 = max(0, int(oy1))
                            gx2 = min(frame_bgr.shape[1] - 1, int(ox2)); gy2 = min(frame_bgr.shape[0] - 1, int(oy2))
                            if gx2 > gx1 and gy2 > gy1:
                                phone_patch = frame_bgr[gy1:gy2, gx1:gx2]
                                hsv = cv2.cvtColor(phone_patch, cv2.COLOR_BGR2HSV)
                                # bright and low saturation pixels -> specular/glare
                                v = hsv[:, :, 2]
                                s = hsv[:, :, 1]
                                glare_mask = (v >= self.phone_glare_v_thresh) & (s <= self.phone_glare_s_thresh)
                                glare_frac = float(glare_mask.sum()) / float(max(1, glare_mask.size))
                                # edge density
                                gray = cv2.cvtColor(phone_patch, cv2.COLOR_BGR2GRAY)
                                edges = cv2.Canny(gray, 80, 160)
                                edge_density = float((edges > 0).sum()) / float(max(1, edges.size))
                                if glare_frac >= self.phone_glare_frac_max and edge_density < self.phone_edge_density_min:
                                    # looks like torch/glare, skip
                                    continue
                        except Exception:
                            pass
                        cx1, cy1, cx2, cy2 = float(x1), float(y1), float(x2), float(y2)
                        ix1, iy1 = max(cx1, ox1), max(cy1, oy1)
                        ix2, iy2 = min(cx2, ox2), min(cy2, oy2)
                        if ix1 < ix2 and iy1 < iy2:
                            phone_near_person = True
                            phone_bbox_crop = (ox1 - x1, oy1 - y1, ox2 - x1, oy2 - y1)
                        for hb in hand_boxes_crop:
                            # Simple IoU in crop space
                            hx1, hy1, hx2, hy2 = hb
                            px1, py1, px2, py2 = phone_bbox_crop if phone_bbox_crop else (0, 0, 0, 0)
                            inter_x1 = max(hx1, px1)
                            inter_y1 = max(hy1, py1)
                            inter_x2 = min(hx2, px2)
                            inter_y2 = min(hy2, py2)
                            iw = max(0.0, inter_x2 - inter_x1)
                            ih = max(0.0, inter_y2 - inter_y1)
                            inter_area = iw * ih
                            if phone_area > 0 and (inter_area / phone_area) >= self.phone_hand_iou_min_frac:
                                held_object = "cell phone"
                                held_object_bbox = [ox1, oy1, ox2, oy2]
                                break

                # Check green flag overlap with this person via hand intersection
                if green_flags:
                    # Convert any flag bbox to crop-space if overlapping person
                    for (fx1, fy1, fx2, fy2) in green_flags:
                        if iou(pb, (fx1, fy1, fx2, fy2)) <= 0.0:
                            continue
                        # Require flag to overlap at least one window region (person signaling out of window)
                        if window_regions:
                            overlaps_window = any(iou((fx1, fy1, fx2, fy2), w) > 0.05 for w in window_regions)
                            if not overlaps_window:
                                continue
                        # crop-space flag
                        fcx1, fcy1, fcx2, fcy2 = fx1 - x1, fy1 - y1, fx2 - x1, fy2 - y1
                        for hb in hand_boxes_crop:
                            hx1, hy1, hx2, hy2 = hb
                            ix1, iy1 = max(hx1, fcx1), max(hy1, fcy1)
                            ix2, iy2 = min(hx2, fcx2), min(hy2, fcy2)
                            if (ix2 - ix1) > 0 and (iy2 - iy1) > 0:
                                per_frame_activities.append({
                                    "person_id": pid,
                                    "person_bbox": [x1, y1, x2, y2],
                                    "object": "signal exchange with flag",
                                    "object_bbox": [fx1, fy1, fx2, fy2],
                                    "holding": True,
                                    "evidence": {"rule": "green_flag_hand_intersection_window"},
                                    "track_id": track_id,
                                })
                                flag_interaction_emitted = True
                                break
                        if flag_interaction_emitted:
                            break

                # Phone inference via ear/mouth proximity, with additional suppressions
                inferred_phone, ev = infer_phone_usage_from_landmarks((0.0, 0.0, float(x2 - x1), float(y2 - y1)), face_bbox_crop, hands_list)
                if inferred_phone:
                    # Suppress when head is strongly down (likely operating/packing) or hands too low in the crop
                    if (head_down_angle is not None) and (head_down_angle >= self.phone_infer_head_down_deg):
                        inferred_phone = False
                        ev = {"rule": "suppressed_head_down"}
                    else:
                        if hands_list:
                            hand_ys = [float(p.get("y", 0.0)) for h in hands_list for p in h.values()]
                            if hand_ys:
                                max_hand_y = max(hand_ys)
                                if (max_hand_y / max(1.0, float(y2 - y1))) > self.phone_infer_max_hand_y_frac:
                                    inferred_phone = False
                                    ev = {"rule": "suppressed_hands_low"}

                # Build per-person activities (phone) — only if we have strong evidence
                if held_object:
                    obj_name = held_object
                    evidence = {"rule": "hand_object_intersection_frac"}
                    per_frame_activities.append({
                        "person_id": pid,
                        "person_bbox": [x1, y1, x2, y2],
                        "object": obj_name,
                        "object_bbox": held_object_bbox,
                        "holding": True,
                        "evidence": evidence,
                        "track_id": track_id,
                    })
                elif inferred_phone:
                    obj_name = "cell phone"
                    evidence = dict(ev)
                    per_frame_activities.append({
                        "person_id": pid,
                        "person_bbox": [x1, y1, x2, y2],
                        "object": obj_name,
                        "object_bbox": None,
                        "holding": True,
                        "evidence": evidence,
                        "track_id": track_id,
                    })

                # Sleep tracking inputs
                # Compute EAR and map to open probability
                ear = None
                open_prob = None
                try:
                    ear = compute_ear(face_res, (frame_bgr.shape[0], frame_bgr.shape[1]))
                    open_prob = eye_open_probability(ear)
                except Exception:
                    ear = None
                    open_prob = None

                head_down_angle = None
                if pose_res and pose_res.pose_landmarks:
                    lm = pose_res.pose_landmarks.landmark
                    try:
                        ls = (lm[11].x * frame_bgr.shape[1], lm[11].y * frame_bgr.shape[0])
                        rs = (lm[12].x * frame_bgr.shape[1], lm[12].y * frame_bgr.shape[0])
                        lh = (lm[23].x * frame_bgr.shape[1], lm[23].y * frame_bgr.shape[0])
                        rh = (lm[24].x * frame_bgr.shape[1], lm[24].y * frame_bgr.shape[0])
                        nose = (lm[0].x * frame_bgr.shape[1], lm[0].y * frame_bgr.shape[0])
                        mid_sh = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
                        mid_hp = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
                        # Head-down heuristic angle between head vector and torso-up
                        torso_up = (mid_sh[0] - mid_hp[0], mid_sh[1] - mid_hp[1])
                        head_vec = (nose[0] - mid_sh[0], nose[1] - mid_sh[1])
                        # angle between normalized vectors
                        def ang(v1, v2):
                            import math
                            def n(v):
                                m = max(1e-6, (v[0] ** 2 + v[1] ** 2) ** 0.5)
                                return (v[0] / m, v[1] / m)
                            v1n = n(v1); v2n = n(v2)
                            c = max(-1.0, min(1.0, v1n[0] * v2n[0] + v1n[1] * v2n[1]))
                            return math.degrees(math.acos(c))
                        head_down_angle = ang(head_vec, torso_up)
                    except Exception:
                        head_down_angle = None
                # Head pitch in degrees; positive is head-down per our heuristic

                engaged = any(a.get("person_id") == pid and a.get("object") == "cell phone" for a in per_frame_activities)

                debug_info = None
                if self.use_advanced_sleep and sleep_decider is not None:
                    # Use advanced decision machine
                    nose_xy = None
                    try:
                        if pose_res and pose_res.pose_landmarks:
                            lm = pose_res.pose_landmarks.landmark
                            nose_xy = (lm[0].x * frame_bgr.shape[1], lm[0].y * frame_bgr.shape[0])
                    except Exception:
                        nose_xy = None
                    emitted = sleep_decider.update(
                        track_id=track_id,
                        ts=float(ts),
                        ear=ear,
                        open_prob=open_prob,
                        head_pitch_deg=head_down_angle,
                        head_point_xy=nose_xy,
                        engaged=engaged,
                    )
                    debug_info = sleep_decider.get_debug(track_id, float(ts))
                    if emitted:
                        per_frame_activities.append({
                            "person_id": pid,
                            "person_bbox": [x1, y1, x2, y2],
                            "object": emitted.get("activity"),
                            "object_bbox": None,
                            "holding": False,
                            "evidence": {"rule": emitted.get("evidence_rule")},
                            "track_id": track_id,
                        })
                else:
                    # Keep existing simple SleepTracker logic
                    eye_open = open_prob if open_prob is not None else None
                    slevents = sleep_tracker.update(
                        track_id=track_id,
                        ts=float(ts),
                        person_bbox=[x1, y1, x2, y2],
                        eye_openness=eye_open,
                        head_down_angle_deg=head_down_angle,
                        engaged=engaged,
                    )
                    for sev in slevents:
                        per_frame_activities.append({
                            "person_id": pid,
                            "person_bbox": [x1, y1, x2, y2],
                            "object": sev.get("activity"),
                            "object_bbox": None,
                            "holding": False,
                            "evidence": {"rule": sev.get("evidence_rule")},
                            "track_id": track_id,
                        })

                # Optional debug overlays saved to output/debug
                if self.save_debug_overlays:
                    lines = []
                    lines.append(f"EAR: {ear:.3f}" if ear is not None else "EAR: None")
                    lines.append(f"open_prob: {open_prob:.2f}" if open_prob is not None else "open_prob: None")
                    lines.append(f"head_pitch_deg: {head_down_angle:.1f}" if head_down_angle is not None else "head_pitch_deg: None")
                    if debug_info:
                        lines.append(f"state: {debug_info.get('state')}")
                        lines.append(f"closed_run_s: {debug_info.get('closed_run_s'):.2f}")
                        lines.append(f"perclos_mw: {debug_info.get('perclos_mw'):.2f}")
                        lines.append(f"low_motion_sw: {debug_info.get('low_motion_sw')}")
                    tag_base = f"frame_{index:06d}_{ts:.2f}s"
                    try:
                        save_debug_overlay(frame_bgr, lines, tag=tag_base, out_dir=os.path.join(os.path.dirname(video_path), "output"))
                    except Exception:
                        pass

                # Heuristic tracker for activities 4,5 (writing, packing)
                # Build frame-space hand boxes for this person from crop-space boxes
                hand_boxes_frame = []
                for hb in hand_boxes_crop:
                    hx1, hy1, hx2, hy2 = hb
                    hand_boxes_frame.append([hx1 + x1, hy1 + y1, hx2 + x1, hy2 + y1])
                # Bag/object boxes from detections (non-person) with filtering per person context
                bag_boxes_frame = []
                # Person geometry
                pw = max(1.0, float(x2 - x1)); ph = max(1.0, float(y2 - y1))
                person_area = pw * ph
                # Define torso/waist vertical band within the person box
                torso_y1 = y1 + int(0.20 * ph)
                torso_y2 = y1 + int(min(1.0, 0.20 + self.bag_torso_band_frac) * ph)
                for (cid, score, ob) in objects:
                    if cid not in (24, 26, 27):
                        continue
                    if float(score) < float(self.bag_min_score):
                        continue
                    if iou(pb, ob) <= 0.0:
                        continue
                    bx1, by1, bx2, by2 = ob
                    ow = max(1.0, float(bx2 - bx1)); oh = max(1.0, float(by2 - by1))
                    bag_area = ow * oh
                    area_frac = bag_area / max(1.0, person_area)
                    if not (self.bag_area_frac_min <= area_frac <= self.bag_area_frac_max):
                        continue
                    # Check if bag intersects torso band
                    ib_y1 = max(by1, torso_y1)
                    ib_y2 = min(by2, torso_y2)
                    if (ib_y2 - ib_y1) <= 0:
                        continue
                    bag_boxes_frame.append([bx1, by1, bx2, by2])
                # Prepare pose points (shoulders/wrists) in frame coords for extension check
                pose_points = {}
                if pose_res and pose_res.pose_landmarks:
                    lm = pose_res.pose_landmarks.landmark
                    def p(idx):
                        return (lm[idx].x * frame_bgr.shape[1], lm[idx].y * frame_bgr.shape[0])
                    try:
                        pose_points = {
                            "left_shoulder": p(11),
                            "right_shoulder": p(12),
                            "left_wrist": p(15),
                            "right_wrist": p(16),
                        }
                    except Exception:
                        pose_points = {}

                act_events = act_tracker.update(
                    track_id=track_id,
                    ts=float(ts),
                    person_bbox=[x1, y1, x2, y2],
                    hand_boxes_frame=hand_boxes_frame,
                    bag_boxes_frame=bag_boxes_frame,
                    pose_points=pose_points,
                )
                for aev in act_events:
                    per_frame_activities.append({
                        "person_id": pid,
                        "person_bbox": [x1, y1, x2, y2],
                        "object": aev.get("activity"),
                        "object_bbox": None,
                        "holding": False,
                        "evidence": {"rule": aev.get("evidence_rule")},
                        "track_id": track_id,
                    })

            # Add group activity if more than two people are present
            try:
                if len(persons) > 2:
                    person_boxes = [b for (_, _, b) in persons]
                    gx1 = int(min(b[0] for b in person_boxes))
                    gy1 = int(min(b[1] for b in person_boxes))
                    gx2 = int(max(b[2] for b in person_boxes))
                    gy2 = int(max(b[3] for b in person_boxes))
                    per_frame_activities.append({
                        "person_id": "group",
                        "person_bbox": [gx1, gy1, gx2, gy2],
                        "object": "more_than_two_people",
                        "object_bbox": None,
                        "holding": False,
                        "evidence": {"rule": "people_count", "count": len(persons)},
                        "track_id": None,
                    })
            except Exception:
                pass

            # Annotate frame if any activity
            # Use filtered person detections for visualization to avoid confusion
            filtered_detections = persons + objects
            tag_base = f"frame_{index:06d}_{ts:.2f}s"
            result_for_annot = {
                "detections": [
                    {"bbox": b, "conf": s, "name": "person" if c == 0 else "object"}
                    for (c, s, b) in filtered_detections
                ],
                "activities": per_frame_activities,
            }
            annotate_and_save(frame_bgr, result_for_annot, tag=tag_base, out_dir=os.path.join(os.path.dirname(video_path), "output"))

            # Convert activities to schema events (write ALL activities)
            # Mapping:
            # 1 micro_sleep, 2 sleep, 3 phone, 4 writing, 5 packing, 7 signal exchange with flag
            def _map_activity(label: str) -> Tuple[int, str]:
                key = (label or "").strip().lower()
                # Common normalizations
                normalized = key.replace("-", " ").replace("_", " ").strip()
                mapping: Dict[str, Tuple[int, str]] = {
                    "micro sleep": (1, "Micro sleep episode"),
                    "micro_sleep": (1, "Micro sleep episode"),
                    "sleep": (2, "Sleeping episode"),
                    "cell phone": (3, "Using cell phone"),
                    "writing": (4, "Writing while moving"),
                    "packing": (5, "Packing"),
                    "signal exchange with flag": (7, "Signal exchange with flag"),
                    "flag exchange": (7, "Signal exchange with flag"),
                    "more than two people": (8, "More than two people detected"),
                    "more_than_two_people": (8, "More than two people detected"),
                    "group": (8, "More than two people detected"),
                }
                if key in mapping:
                    return mapping[key]
                if normalized in mapping:
                    return mapping[normalized]
                # Fallback: unknown activity; keep but mark as 0 with readable description
                return 0, (normalized.title() if normalized else "Unknown activity")

            for act in per_frame_activities:
                obj = str(act.get("object", "")).lower()
                activity_type, des = _map_activity(obj)
                # If this is the group-count event, override description with actual count
                if obj in ("more than two people", "more_than_two_people", "group"):
                    try:
                        ev = act.get("evidence") or {}
                        cnt = ev.get("count")
                        if isinstance(cnt, int) and cnt >= 0:
                            des = f"{cnt} people detected"
                    except Exception:
                        pass
                event = ActivityEvent(
                    tripId=self.trip_id,
                    activityType=activity_type,
                    des=des,
                    objectType=obj,
                    fileUrl=video_path,
                    fileDuration=file_duration,
                    activityStartTime=f"{ts:.2f}",
                    crewName=self.crew_name,
                    crewId=self.crew_id,
                    crewRole=self.crew_role,
                    date=date_str,
                    time=time_str,
                    filename=filename,
                    peopleCount=len(persons),
                    evidence=act.get("evidence"),
                    activityImage=f"{tag_base}_activity.jpg",
                )
                events.append(event)

            processed += 1
            if callable(progress_cb):
                try:
                    progress_cb({"processed": processed, "total": expected_total})
                except Exception:
                    pass

        # finalize sleep episodes (if any open)
        pending = sleep_tracker.finalize()
        # No specific timestamp here; episodes already appended on close per frame above
        logger.info(f"[Pipeline] Processed {processed} sampled frames from {filename} (sample_fps={self.sample_fps}, duration={file_duration})")
        if callable(progress_cb):
            try:
                progress_cb({"processed": processed, "total": expected_total, "done": True})
            except Exception:
                pass
        return events
