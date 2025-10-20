from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import os
import cv2
import numpy as np

from app.models.activity_event import ActivityEvent
from app.services.yolo_service import YoloService
from app.services.mediapipe_service import MediaPipeService
from app.utils.ocr_utils import OcrUtils
from app.utils.video_utils import sample_video_frames, get_video_duration_str, get_video_filename, get_expected_sampled_frames, get_expected_sampled_frames_in_range
from app.utils.geometry import iou
from app.utils.phone_logic import infer_phone_usage_from_landmarks
from app.services.antenna_refiner import AntennaRefiner
from app.utils.annotate import annotate_and_save
from app.utils.trackers import SimpleTracker, SleepTracker, ActivityHeuristicTracker
from app.utils.eye_metrics import compute_ear, eye_open_probability
from app.utils.sleep_decision import SleepDecisionMachine, SleepDecisionConfig
from app.utils.debug_overlay import save_debug_overlay
from app.utils.flag_utils import detect_green_flags
from app.utils.window_utils import detect_window_regions
from loguru import logger
from app.utils.clip_utils import extract_clip


@dataclass
class ActivityPipeline:
    """Main activity detection pipeline.

    Orchestrates YOLO detections, MediaPipe landmarks, OCR (optional), sleep logic,
    and heuristic trackers to emit structured `ActivityEvent` records.
    """
    trip_id: str
    crew_name: str
    crew_id: str
    crew_role: int
    yolo_weights: str
    sample_fps: int = 1
    enable_ocr: bool = True
    verbose: bool = False
    max_frames: int = 0
    # Micro-batching knob (used in next step for YOLO batching)
    yolo_batch: int = 1
    # Sleep thresholds
    sleep_eye_thresh: float = 0.18
    sleep_headdown_deg: float = 100.0
    sleep_micro_max_min: float = 0.25
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
    # Tunable advanced sleep config (defaults aimed at higher sensitivity)
    sleep_cfg_short_window_s: float = 4.0
    sleep_cfg_mid_window_s: float = 30.0
    sleep_cfg_long_window_s: float = 120.0
    sleep_cfg_smoothing_alpha: float = 0.5
    sleep_cfg_eye_closed_run_s: float = 2.2
    sleep_cfg_perclos_drowsy_thresh: float = 0.35
    sleep_cfg_perclos_sleep_thresh: float = 0.75
    sleep_cfg_head_pitch_down_deg: float = 20.0
    sleep_cfg_head_neutral_deg: float = 12.0
    sleep_cfg_hold_transition_s: float = 1.0
    sleep_cfg_recovery_hold_s: float = 2.0
    sleep_cfg_open_prob_closed_thresh: float = 0.45
    sleep_cfg_no_eye_head_down_deg: float = 32.0
    # Phone inference robustness
    phone_hand_iou_min_frac: float = 0.15
    phone_infer_min_face_frac: float = 0.02
    phone_infer_suppress_head_down: bool = True
    phone_infer_head_down_deg: float = 35.0
    phone_infer_max_hand_y_frac: float = 0.65
    # Packing detection config (scalable knobs)
    pack_iou_overlap_thresh: float = 0.08
    pack_window_s: float = 2.0
    bag_min_score: float = 0.30
    bag_area_frac_min: float = 0.005
    bag_area_frac_max: float = 0.35
    bag_torso_band_frac: float = 0.70
    # Writing detection config (scalable)
    write_window_s: float = 3.0
    write_min_path_px: float = 100.0
    write_max_radius_px: float = 25.0
    write_lap_band_frac: float = 0.40
    # Require a visible surface (book/paper) to emit 'writing'
    write_require_surface: bool = True
    write_ocr_min_chars: int = 14

    def _map_activity_label(self, label: str) -> Tuple[int, str]:
        key = (label or "").strip().lower()
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
        return 0, (normalized.title() if normalized else "Unknown activity")

    def process_video(self, video_path: str, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[ActivityEvent]:
        """Process an entire video at `sample_fps` and return activity events.

        Progress is optionally reported via `progress_cb({"processed": n, "total": m, "done": bool})`.
        """
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
        antenna_refiner = AntennaRefiner(yolo_weights=self.yolo_weights, use_heuristic=True)
        ocr = OcrUtils() if self.enable_ocr else None

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
        sleep_decider = SleepDecisionMachine(SleepDecisionConfig(
            short_window_s=self.sleep_cfg_short_window_s,
            mid_window_s=self.sleep_cfg_mid_window_s,
            long_window_s=self.sleep_cfg_long_window_s,
            smoothing_alpha=self.sleep_cfg_smoothing_alpha,
            eye_closed_run_s=self.sleep_cfg_eye_closed_run_s,
            perclos_drowsy_thresh=self.sleep_cfg_perclos_drowsy_thresh,
            perclos_sleep_thresh=self.sleep_cfg_perclos_sleep_thresh,
            head_pitch_down_deg=self.sleep_cfg_head_pitch_down_deg,
            head_neutral_deg=self.sleep_cfg_head_neutral_deg,
            hold_transition_s=self.sleep_cfg_hold_transition_s,
            recovery_hold_s=self.sleep_cfg_recovery_hold_s,
            open_prob_closed_thresh=self.sleep_cfg_open_prob_closed_thresh,
            head_down_micro_fallback=True,
            head_down_micro_deg=max(self.sleep_cfg_head_pitch_down_deg, 40.0),
            ear_high_weird_threshold=0.42,
            no_eye_head_down_deg=self.sleep_cfg_no_eye_head_down_deg,
        )) if self.use_advanced_sleep else None

        processed = 0
        try:
            expected_total = int(get_expected_sampled_frames(video_path, self.sample_fps))
            if self.max_frames and self.max_frames > 0:
                expected_total = min(expected_total, int(self.max_frames))
        except Exception:
            expected_total = 0
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
            green_flags = detect_green_flags(frame_bgr)
            window_regions = detect_window_regions(frame_bgr)
            if ocr is not None:
                try:
                    date_str, time_str = ocr.extract_date_time(frame_bgr)
                except Exception:
                    date_str, time_str = "", ""
            else:
                date_str, time_str = "", ""

            H, W = frame_bgr.shape[0], frame_bgr.shape[1]
            frame_area = float(H * W)
            min_area = max(1.0, self.person_min_area_frac * frame_area)
            person_candidates = [
                (cid, score, box)
                for (cid, score, box) in detections
                if cid == 0 and float(score) >= float(self.person_min_conf)
                and max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1])) >= min_area
            ]
            person_candidates.sort(key=lambda d: float(d[1]), reverse=True)
            persons: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
            for det in person_candidates:
                _, _, b = det
                if all(iou(b, kept[2]) < self.person_nms_iou for kept in persons):
                    persons.append(det)

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
                if validated:
                    persons = validated
            objects = [(cid, score, box) for (cid, score, box) in detections if cid != 0]

            person_boxes_np = [np.array(b, dtype=float) for (_, _, b) in persons]
            track_ids = tracker.assign(person_boxes_np, float(ts)) if person_boxes_np else []

            per_frame_activities: List[Dict[str, Any]] = []

            for pid, (track_id, (_, _, pb)) in enumerate(zip(track_ids, persons), start=1):
                x1, y1, x2, y2 = map(int, pb)
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame_bgr.shape[1] - 1, x2); y2 = min(frame_bgr.shape[0] - 1, y2)
                crop = frame_bgr[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                hands_res = mp_out.get('hands')
                face_res = mp_out.get('face')
                pose_res = mp_out.get('pose')

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

                face_bbox_crop = None
                if face_res and face_res.multi_face_landmarks:
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

                hand_boxes_crop = []
                for hand in hands_list:
                    xs = [p["x"] for p in hand.values()]
                    ys = [p["y"] for p in hand.values()]
                    if xs and ys:
                        hx1, hy1, hx2, hy2 = min(xs), min(ys), max(xs), max(ys)
                        pad = 10.0
                        hand_boxes_crop.append((hx1 - pad, hy1 - pad, hx2 + pad, hy2 + pad))

                phone_near_person = False
                phone_bbox_crop = None
                held_object = None
                held_object_bbox = None
                flag_interaction_emitted = False

                for (cid, score, ob) in objects:
                    if iou(pb, ob) <= 0.0:
                        continue
                    name = "phone" if cid == 67 else "other"
                    if cid == 67:
                        if float(score) < float(self.phone_min_conf):
                            continue
                        ox1, oy1, ox2, oy2 = ob
                        pw = max(1.0, float(x2 - x1)); ph = max(1.0, float(y2 - y1))
                        person_area = pw * ph
                        ow = max(1.0, float(ox2 - ox1)); oh = max(1.0, float(oy2 - oy1))
                        phone_area = ow * oh
                        area_frac = phone_area / max(1.0, person_area)
                        ar = ow / oh
                        if area_frac > self.phone_max_area_person_frac or ar < self.phone_ar_min or ar > self.phone_ar_max:
                            continue
                        try:
                            gx1 = max(0, int(ox1)); gy1 = max(0, int(oy1))
                            gx2 = min(frame_bgr.shape[1] - 1, int(ox2)); gy2 = min(frame_bgr.shape[0] - 1, int(oy2))
                            if gx2 > gx1 and gy2 > gy1:
                                phone_patch = frame_bgr[gy1:gy2, gx1:gx2]
                                hsv = cv2.cvtColor(phone_patch, cv2.COLOR_BGR2HSV)
                                v = hsv[:, :, 2]
                                s = hsv[:, :, 1]
                                glare_mask = (v >= self.phone_glare_v_thresh) & (s <= self.phone_glare_s_thresh)
                                glare_frac = float(glare_mask.sum()) / float(max(1, glare_mask.size))
                                gray = cv2.cvtColor(phone_patch, cv2.COLOR_BGR2GRAY)
                                edges = cv2.Canny(gray, 80, 160)
                                edge_density = float((edges > 0).sum()) / float(max(1, edges.size))
                                if glare_frac >= self.phone_glare_frac_max and edge_density < self.phone_edge_density_min:
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

                if green_flags:
                    for (fx1, fy1, fx2, fy2) in green_flags:
                        if iou(pb, (fx1, fy1, fx2, fy2)) <= 0.0:
                            continue
                        if window_regions:
                            overlaps_window = any(iou((fx1, fy1, fx2, fy2), w) > 0.05 for w in window_regions)
                            if not overlaps_window:
                                continue
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

                inferred_phone, ev = infer_phone_usage_from_landmarks((0.0, 0.0, float(x2 - x1), float(y2 - y1)), face_bbox_crop, hands_list)
                if inferred_phone:
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
                            torso_up = (mid_sh[0] - mid_hp[0], mid_sh[1] - mid_hp[1])
                            head_vec = (nose[0] - mid_sh[0], nose[1] - mid_sh[1])
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

                if held_object:
                    obj_name = held_object
                    # If object detected as cell phone, refine with antenna check to reclassify as walkie-talkie
                    if obj_name == "cell phone" and held_object_bbox is not None:
                        try:
                            has_ant, ant_ev = antenna_refiner.has_antenna(frame_bgr, held_object_bbox)
                            if has_ant:
                                obj_name = "walkie_talkie"
                                evidence = {"rule": "antenna_refiner", **(ant_ev or {})}
                                # Skip logging walkie-talkie per requirement
                                pass
                        except Exception:
                            pass
                    evidence = {"rule": "hand_object_intersection_frac"}
                    if obj_name != "walkie_talkie":
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
                    obj_name = str(ev.get("device_type", "cell phone")).lower() if isinstance(ev, dict) else "cell phone"
                    evidence = dict(ev) if isinstance(ev, dict) else {}
                    # If inferred as cell phone, try antenna refinement using a proxy bbox around mouth/hand if available is not present
                    if obj_name == "cell phone":
                        try:
                            # Build a rough bbox around the face mouth zone if present in evidence; fallback to person bbox
                            proxy_bbox = None
                            try:
                                zone = evidence.get("zone")
                                if zone and "mouth" in str(zone):
                                    # approximate small box near face center; using person crop since we don't have face bbox absolute here
                                    cx = (x1 + x2) / 2.0
                                    cy = y1 + 0.25 * (y2 - y1)
                                    w = 0.12 * (x2 - x1)
                                    h = 0.18 * (y2 - y1)
                                    proxy_bbox = [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]
                            except Exception:
                                proxy_bbox = None
                            if proxy_bbox is None:
                                proxy_bbox = [int(x1), int(y1), int(x2), int(y2)]
                            has_ant, ant_ev = antenna_refiner.has_antenna(frame_bgr, proxy_bbox)
                            if has_ant:
                                obj_name = "walkie_talkie"
                                evidence = {"rule": "antenna_refiner", **(ant_ev or {})}
                        except Exception:
                            pass
                    if obj_name != "walkie_talkie":
                        per_frame_activities.append({
                            "person_id": pid,
                            "person_bbox": [x1, y1, x2, y2],
                            "object": obj_name,
                            "object_bbox": None,
                            "holding": True,
                            "evidence": evidence,
                            "track_id": track_id,
                        })

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
                        torso_up = (mid_sh[0] - mid_hp[0], mid_sh[1] - mid_hp[1])
                        head_vec = (nose[0] - mid_sh[0], nose[1] - mid_sh[1])
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

                engaged = any(a.get("person_id") == pid and a.get("object") == "cell phone" for a in per_frame_activities)

                if self.use_advanced_sleep and sleep_decider is not None:
                    nose_xy = None
                    try:
                        if pose_res and pose_res.pose_landmarks:
                            lm = pose_res.pose_landmarks.landmark
                            nose_xy = (lm[0].x * frame_bgr.shape[1], lm[0].y * frame_bgr.shape[0])
                    except Exception:
                        nose_xy = None
                    head_point_xy = nose_xy
                    if head_point_xy is None and face_bbox_crop:
                        try:
                            fx1, fy1, fx2, fy2 = face_bbox_crop
                            head_point_xy = (x1 + (fx1 + fx2) / 2.0, y1 + (fy1 + fy2) / 2.0)
                        except Exception:
                            head_point_xy = None
                    if head_point_xy is None:
                        head_point_xy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                    emitted = sleep_decider.update(
                        track_id=track_id,
                        ts=float(ts),
                        ear=ear,
                        open_prob=open_prob,
                        head_pitch_deg=head_down_angle,
                        head_point_xy=head_point_xy,
                        engaged=engaged,
                    )
                    debug_info = sleep_decider.get_debug(track_id, float(ts))
                    if emitted:
                        activity_label = str(emitted.get("activity", ""))
                        if activity_label == "micro_sleep":
                            conf = float(emitted.get("confidence", 0.0))
                            rule = str(emitted.get("rule", ""))
                            closed_run = float(emitted.get("closed_run_s", 0.0))
                            allow = False
                            if rule == "eye_path":
                                allow = (conf >= 0.60) and (closed_run >= self.sleep_cfg_eye_closed_run_s)
                            else:
                                allow = conf >= 0.80
                            if allow:
                                per_frame_activities.append({
                                    "person_id": pid,
                                    "person_bbox": [x1, y1, x2, y2],
                                    "object": activity_label,
                                    "object_bbox": None,
                                    "holding": False,
                                    "evidence": {"rule": emitted.get("evidence_rule")},
                                    "track_id": track_id,
                                })
                        else:
                            per_frame_activities.append({
                                "person_id": pid,
                                "person_bbox": [x1, y1, x2, y2],
                                "object": activity_label,
                                "object_bbox": None,
                                "holding": False,
                                "evidence": {"rule": emitted.get("evidence_rule")},
                                "track_id": track_id,
                            })
                    else:
                        if debug_info and debug_info.get("state") == "sleep":
                            per_frame_activities.append({
                                "person_id": pid,
                                "person_bbox": [x1, y1, x2, y2],
                                "object": "sleep",
                                "object_bbox": None,
                                "holding": False,
                                "evidence": {"rule": "state_hold"},
                                "track_id": track_id,
                            })
                else:
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
                            # Carry true episode timing when available from SleepTracker
                            "event_start_ts": sev.get("start_ts"),
                            "event_end_ts": sev.get("end_ts"),
                        })

                if self.save_debug_overlays:
                    lines = []
                    lines.append(f"EAR: {ear:.3f}" if ear is not None else "EAR: None")
                    lines.append(f"open_prob: {open_prob:.2f}" if open_prob is not None else "open_prob: None")
                    lines.append(f"head_pitch_deg: {head_down_angle:.1f}" if head_down_angle is not None else "head_pitch_deg: None")
                    tag_base = f"frame_{index:06d}_{ts:.2f}s"
                    try:
                        save_debug_overlay(frame_bgr, lines, tag=tag_base, out_dir=os.path.join(os.path.dirname(video_path), "output"))
                    except Exception:
                        pass

                hand_boxes_frame = []
                for hb in hand_boxes_crop:
                    hx1, hy1, hx2, hy2 = hb
                    hand_boxes_frame.append([hx1 + x1, hy1 + y1, hx2 + x1, hy2 + y1])
                bag_boxes_frame = []
                pw = max(1.0, float(x2 - x1)); ph = max(1.0, float(y2 - y1))
                person_area = pw * ph
                torso_y1 = y1 + int(0.20 * ph)
                torso_y2 = y1 + int(min(1.0, 0.20 + self.bag_torso_band_frac) * ph)
                for (cid, score, ob) in objects:
                    if cid not in (24, 26, 28):
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
                    ib_y1 = max(by1, torso_y1)
                    ib_y2 = min(by2, torso_y2)
                    if (ib_y2 - ib_y1) <= 0:
                        continue
                    bag_boxes_frame.append([bx1, by1, bx2, by2])

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
                    label = str(aev.get("activity", "")).lower()
                    evidence_rule = str(aev.get("evidence_rule", ""))

                    allow_emit = True
                    evidence = {"rule": evidence_rule}

                    # Gate 'writing' on presence of a surface (book or OCR text) in lap area
                    if label == "writing" and bool(self.write_require_surface):
                        ph = max(1.0, float(y2 - y1))
                        lap_y1 = y1 + int((1.0 - float(self.write_lap_band_frac)) * ph)
                        lap_rect = (float(x1), float(lap_y1), float(x2), float(y2))

                        # Gate 1: YOLO 'book' (COCO id 73) overlapping lap band
                        has_book_surface = any(
                            (cid == 73) and (iou(pb, ob) > 0.0) and (iou(lap_rect, ob) > 0.05)
                            for (cid, score, ob) in objects
                        )

                        # Gate 2: OCR finds sufficient text in lap band (proxy for paper)
                        has_text_surface = False
                        if (not has_book_surface) and (ocr is not None):
                            try:
                                lap_crop = frame_bgr[lap_y1:y2, x1:x2]
                                txt = (ocr.extract_text(lap_crop) or "").strip()
                                has_text_surface = (len(txt) >= int(self.write_ocr_min_chars))
                            except Exception:
                                has_text_surface = False

                        allow_emit = bool(has_book_surface or has_text_surface)
                        if allow_emit:
                            evidence = {"rule": "writing_surface_present_book" if has_book_surface else "writing_surface_present_ocr"}
                        else:
                            # Suppress false positive (e.g., gear/lever motion without surface)
                            evidence = {"rule": "writing_suppressed_no_surface"}

                    if allow_emit:
                        per_frame_activities.append({
                            "person_id": pid,
                            "person_bbox": [x1, y1, x2, y2],
                            "object": label,
                            "object_bbox": None,
                            "holding": False,
                            "evidence": evidence,
                            "track_id": track_id,
                        })

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

            filtered_detections = persons + objects
            tag_base = f"frame_{index:06d}_{ts:.2f}s"
            result_for_annot = {
                "detections": [
                    {"bbox": b, "conf": s, "name": "person" if c == 0 else "object"}
                    for (c, s, b) in filtered_detections
                ],
                "activities": per_frame_activities,
            }
            # Write overlays/images only when debugging or when activities are present
            if self.save_debug_overlays or per_frame_activities:
                try:
                    annotate_and_save(frame_bgr, result_for_annot, tag=tag_base, out_dir=os.path.join(os.path.dirname(video_path), "output"))
                except Exception:
                    pass

            # Use shared mapping helper
            def _map_activity(label: str) -> Tuple[int, str]:
                return self._map_activity_label(label)

            for act in per_frame_activities:
                try:
                    ev_tmp = act.get("evidence") or {}
                    if ev_tmp.get("rule") == "state_hold":
                        continue
                except Exception:
                    pass
                obj = str(act.get("object", "")).lower()
                activity_type, des = _map_activity(obj)
                if obj in ("more than two people", "more_than_two_people", "group"):
                    try:
                        ev = act.get("evidence") or {}
                        cnt = ev.get("count")
                        if isinstance(cnt, int) and cnt >= 0:
                            des = f"{cnt} people detected"
                    except Exception:
                        pass
                clip_filename = None
                try:
                    out_dir = os.path.join(os.path.dirname(video_path), "output")
                    clip_filename = extract_clip(video_path, float(ts), out_dir, tag_base, duration_s=4.0)
                except Exception:
                    clip_filename = None

                # Resolve start/end times: prefer tracker-provided values if present
                _ev_start = act.get("event_start_ts")
                _ev_end = act.get("event_end_ts")
                _start_ts = float(_ev_start) if _ev_start is not None else float(ts)
                _end_ts = float(_ev_end) if _ev_end is not None else (float(_start_ts) + 4.0)

                event = ActivityEvent(
                    tripId=self.trip_id,
                    activityType=activity_type,
                    des=des,
                    objectType=obj,
                    fileUrl=video_path,
                    fileDuration=file_duration,
                    activityStartTime=f"{_start_ts:.2f}",
                    activityEndTime=f"{_end_ts:.2f}",
                    crewName=self.crew_name,
                    crewId=self.crew_id,
                    crewRole=self.crew_role,
                    date=date_str,
                    time=time_str,
                    filename=filename,
                    peopleCount=len(persons),
                    evidence=act.get("evidence"),
                    activityImage=f"{tag_base}_activity.jpg",
                    activityClip=clip_filename,
                )
                events.append(event)

            processed += 1
            if callable(progress_cb):
                try:
                    progress_cb({"processed": processed, "total": expected_total})
                except Exception:
                    pass

        pending = sleep_tracker.finalize()
        logger.info(f"[Pipeline] Processed {processed} sampled frames from {filename} (sample_fps={self.sample_fps}, duration={file_duration})")
        if callable(progress_cb):
            try:
                progress_cb({"processed": processed, "total": expected_total, "done": True})
            except Exception:
                pass
        return events


    def process_video_range(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[ActivityEvent]:
        """Process only frames in [start_frame, end_frame) while honoring sampling.

        Frames are sampled at `self.sample_fps` using the same logic as `sample_video_frames`,
        constrained to the absolute frame index range [start_frame, end_frame).
        """
        if self.verbose:
            logger.debug("[Pipeline:Range] Initializing services (YOLO, MediaPipe, OCR)...")
        try:
            import os as _os
            _yolo_conf = float(_os.getenv("YOLO_CONF", "0.25"))
            _yolo_iou = float(_os.getenv("YOLO_IOU", "0.45"))
        except Exception:
            _yolo_conf, _yolo_iou = 0.25, 0.45
        yolo = YoloService(self.yolo_weights, conf=_yolo_conf, iou=_yolo_iou)
        mp_service = MediaPipeService()
        antenna_refiner = AntennaRefiner(yolo_weights=self.yolo_weights, use_heuristic=True)
        ocr = OcrUtils() if self.enable_ocr else None

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
        sleep_decider = SleepDecisionMachine(SleepDecisionConfig(
            short_window_s=self.sleep_cfg_short_window_s,
            mid_window_s=self.sleep_cfg_mid_window_s,
            long_window_s=self.sleep_cfg_long_window_s,
            smoothing_alpha=self.sleep_cfg_smoothing_alpha,
            eye_closed_run_s=self.sleep_cfg_eye_closed_run_s,
            perclos_drowsy_thresh=self.sleep_cfg_perclos_drowsy_thresh,
            perclos_sleep_thresh=self.sleep_cfg_perclos_sleep_thresh,
            head_pitch_down_deg=self.sleep_cfg_head_pitch_down_deg,
            head_neutral_deg=self.sleep_cfg_head_neutral_deg,
            hold_transition_s=self.sleep_cfg_hold_transition_s,
            recovery_hold_s=self.sleep_cfg_recovery_hold_s,
            open_prob_closed_thresh=self.sleep_cfg_open_prob_closed_thresh,
            head_down_micro_fallback=True,
            head_down_micro_deg=max(self.sleep_cfg_head_pitch_down_deg, 40.0),
            ear_high_weird_threshold=0.42,
            no_eye_head_down_deg=self.sleep_cfg_no_eye_head_down_deg,
        )) if self.use_advanced_sleep else None

        # Expected total sampled frames within the range (shared util)
        processed = 0
        expected_total = get_expected_sampled_frames_in_range(video_path, self.sample_fps, start_frame, end_frame)
        if self.max_frames and self.max_frames > 0:
            expected_total = min(expected_total, int(self.max_frames))
        if callable(progress_cb):
            try:
                progress_cb({"processed": processed, "total": expected_total})
            except Exception:
                pass

        # Determine sampling step consistent with sample_video_frames for actual iteration
        cap_meta = cv2.VideoCapture(video_path)
        if not cap_meta.isOpened():
            return []
        native_fps = cap_meta.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(native_fps / max(1, self.sample_fps))))
        cap_meta.release()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        sampled_idx = 0

        # Compute the first sampled frame aligned to "step" inside [start_frame, end_frame)
        start_frame_i = max(0, int(start_frame))
        end_frame_i = max(0, int(end_frame))
        first_sample = ((start_frame_i + step - 1) // step) * step
        if first_sample >= end_frame_i:
            cap.release()
            return []

        # Micro-batching: accumulate sampled frames and run YOLO in batches
        yolo_bs = getattr(self, "yolo_batch", 1)
        try:
            env_bs = os.getenv("YOLO_BATCH", "").strip()
            if env_bs:
                yolo_bs = max(1, int(float(env_bs)))
        except Exception:
            pass

        def _process_batch(batch_frames, batch_meta):
            nonlocal processed, sampled_idx
            if not batch_frames:
                return
            try:
                detections_batched = yolo.detect_batch(batch_frames) if yolo_bs and yolo_bs > 1 else [yolo.detect(img) for img in batch_frames]
            except Exception:
                detections_batched = [yolo.detect(img) for img in batch_frames]

            for (frame_idx, ts), frame_bgr, detections in zip(batch_meta, batch_frames, detections_batched):
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_out = mp_service.process(frame_rgb)
                green_flags = detect_green_flags(frame_bgr)
                window_regions = detect_window_regions(frame_bgr)
                if ocr is not None:
                    try:
                        date_str, time_str = ocr.extract_date_time(frame_bgr)
                    except Exception:
                        date_str, time_str = "", ""
                else:
                    date_str, time_str = "", ""

                    H, W = frame_bgr.shape[0], frame_bgr.shape[1]
                    frame_area = float(H * W)
                    min_area = max(1.0, self.person_min_area_frac * frame_area)
                    person_candidates = [
                        (cid, score, box)
                        for (cid, score, box) in detections
                        if cid == 0 and float(score) >= float(self.person_min_conf)
                        and max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1])) >= min_area
                    ]
                    person_candidates.sort(key=lambda d: float(d[1]), reverse=True)
                    persons: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
                    for det in person_candidates:
                        _, _, b = det
                        if all(iou(b, kept[2]) < self.person_nms_iou for kept in persons):
                            persons.append(det)

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
                        if validated:
                            persons = validated
                    objects = [(cid, score, box) for (cid, score, box) in detections if cid != 0]

                    person_boxes_np = [np.array(b, dtype=float) for (_, _, b) in persons]
                    track_ids = tracker.assign(person_boxes_np, float(ts)) if person_boxes_np else []

                    per_frame_activities: List[Dict[str, Any]] = []

                    for pid, (track_id, (_, _, pb)) in enumerate(zip(track_ids, persons), start=1):
                        x1, y1, x2, y2 = map(int, pb)
                        x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame_bgr.shape[1] - 1, x2); y2 = min(frame_bgr.shape[0] - 1, y2)
                        crop = frame_bgr[y1:y2, x1:x2]
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                        hands_res = mp_out.get('hands')
                        face_res = mp_out.get('face')
                        pose_res = mp_out.get('pose')

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

                        face_bbox_crop = None
                        if face_res and face_res.multi_face_landmarks:
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

                        hand_boxes_crop = []
                        for hand in hands_list:
                            xs = [p["x"] for p in hand.values()]
                            ys = [p["y"] for p in hand.values()]
                            if xs and ys:
                                hx1, hy1, hx2, hy2 = min(xs), min(ys), max(xs), max(ys)
                                pad = 10.0
                                hand_boxes_crop.append((hx1 - pad, hy1 - pad, hx2 + pad, hy2 + pad))

                        phone_near_person = False
                        phone_bbox_crop = None
                        held_object = None
                        held_object_bbox = None
                        flag_interaction_emitted = False

                        for (cid, score, ob) in objects:
                            if iou(pb, ob) <= 0.0:
                                continue
                            name = "phone" if cid == 67 else "other"
                            if cid == 67:
                                if float(score) < float(self.phone_min_conf):
                                    continue
                                ox1, oy1, ox2, oy2 = ob
                                pw = max(1.0, float(x2 - x1)); ph = max(1.0, float(y2 - y1))
                                person_area = pw * ph
                                ow = max(1.0, float(ox2 - ox1)); oh = max(1.0, float(oy2 - oy1))
                                phone_area = ow * oh
                                area_frac = phone_area / max(1.0, person_area)
                                ar = ow / oh
                                if area_frac > self.phone_max_area_person_frac or ar < self.phone_ar_min or ar > self.phone_ar_max:
                                    continue
                                try:
                                    gx1 = max(0, int(ox1)); gy1 = max(0, int(oy1))
                                    gx2 = min(frame_bgr.shape[1] - 1, int(ox2)); gy2 = min(frame_bgr.shape[0] - 1, int(oy2))
                                    if gx2 > gx1 and gy2 > gy1:
                                        phone_patch = frame_bgr[gy1:gy2, gx1:gx2]
                                        hsv = cv2.cvtColor(phone_patch, cv2.COLOR_BGR2HSV)
                                        v = hsv[:, :, 2]
                                        s = hsv[:, :, 1]
                                        glare_mask = (v >= self.phone_glare_v_thresh) & (s <= self.phone_glare_s_thresh)
                                        glare_frac = float(glare_mask.sum()) / float(max(1, glare_mask.size))
                                        gray = cv2.cvtColor(phone_patch, cv2.COLOR_BGR2GRAY)
                                        edges = cv2.Canny(gray, 80, 160)
                                        edge_density = float((edges > 0).sum()) / float(max(1, edges.size))
                                        if glare_frac >= self.phone_glare_frac_max and edge_density < self.phone_edge_density_min:
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

                        if green_flags:
                            for (fx1, fy1, fx2, fy2) in green_flags:
                                if iou(pb, (fx1, fy1, fx2, fy2)) <= 0.0:
                                    continue
                                if window_regions:
                                    overlaps_window = any(iou((fx1, fy1, fx2, fy2), w) > 0.05 for w in window_regions)
                                    if not overlaps_window:
                                        continue
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

                        inferred_phone, ev = infer_phone_usage_from_landmarks((0.0, 0.0, float(x2 - x1), float(y2 - y1)), face_bbox_crop, hands_list)
                        if inferred_phone:
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
                                    torso_up = (mid_sh[0] - mid_hp[0], mid_sh[1] - mid_hp[1])
                                    head_vec = (nose[0] - mid_sh[0], nose[1] - mid_sh[1])
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

                        if held_object:
                            obj_name = held_object
                            if obj_name == "cell phone" and held_object_bbox is not None:
                                try:
                                    has_ant, ant_ev = antenna_refiner.has_antenna(frame_bgr, held_object_bbox)
                                    if has_ant:
                                        obj_name = "walkie_talkie"
                                        evidence = {"rule": "antenna_refiner", **(ant_ev or {})}
                                        pass
                                except Exception:
                                    pass
                            evidence = {"rule": "hand_object_intersection_frac"}
                            if obj_name != "walkie_talkie":
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
                            obj_name = str(ev.get("device_type", "cell phone")).lower() if isinstance(ev, dict) else "cell phone"
                            evidence = dict(ev) if isinstance(ev, dict) else {}
                            if obj_name == "cell phone":
                                try:
                                    proxy_bbox = [int(x1), int(y1), int(x2), int(y2)]
                                    has_ant, ant_ev = antenna_refiner.has_antenna(frame_bgr, proxy_bbox)
                                    if has_ant:
                                        obj_name = "walkie_talkie"
                                        evidence = {"rule": "antenna_refiner", **(ant_ev or {})}
                                except Exception:
                                    pass
                            if obj_name != "walkie_talkie":
                                per_frame_activities.append({
                                    "person_id": pid,
                                    "person_bbox": [x1, y1, x2, y2],
                                    "object": obj_name,
                                    "object_bbox": None,
                                    "holding": True,
                                    "evidence": evidence,
                                    "track_id": track_id,
                                })

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
                                torso_up = (mid_sh[0] - mid_hp[0], mid_sh[1] - mid_hp[1])
                                head_vec = (nose[0] - mid_sh[0], nose[1] - mid_sh[1])
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

                        engaged = any(a.get("person_id") == pid and a.get("object") == "cell phone" for a in per_frame_activities)

                        if self.use_advanced_sleep and sleep_decider is not None:
                            nose_xy = None
                            try:
                                if pose_res and pose_res.pose_landmarks:
                                    lm = pose_res.pose_landmarks.landmark
                                    nose_xy = (lm[0].x * frame_bgr.shape[1], lm[0].y * frame_bgr.shape[0])
                            except Exception:
                                nose_xy = None
                            head_point_xy = nose_xy
                            if head_point_xy is None and face_bbox_crop:
                                try:
                                    fx1, fy1, fx2, fy2 = face_bbox_crop
                                    head_point_xy = (x1 + (fx1 + fx2) / 2.0, y1 + (fy1 + fy2) / 2.0)
                                except Exception:
                                    head_point_xy = None
                            if head_point_xy is None:
                                head_point_xy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                            emitted = sleep_decider.update(
                                track_id=track_id,
                                ts=float(ts),
                                ear=ear,
                                open_prob=open_prob,
                                head_pitch_deg=head_down_angle,
                                head_point_xy=head_point_xy,
                                engaged=engaged,
                            )
                            debug_info = sleep_decider.get_debug(track_id, float(ts))
                            if emitted:
                                activity_label = str(emitted.get("activity", ""))
                                if activity_label == "micro_sleep":
                                    conf = float(emitted.get("confidence", 0.0))
                                    rule = str(emitted.get("rule", ""))
                                    closed_run = float(emitted.get("closed_run_s", 0.0))
                                    allow = False
                                    if rule == "eye_path":
                                        allow = (conf >= 0.60) and (closed_run >= self.sleep_cfg_eye_closed_run_s)
                                    else:
                                        allow = conf >= 0.80
                                    if allow:
                                        per_frame_activities.append({
                                            "person_id": pid,
                                            "person_bbox": [x1, y1, x2, y2],
                                            "object": activity_label,
                                            "object_bbox": None,
                                            "holding": False,
                                            "evidence": {"rule": emitted.get("evidence_rule")},
                                            "track_id": track_id,
                                        })
                                else:
                                    per_frame_activities.append({
                                        "person_id": pid,
                                        "person_bbox": [x1, y1, x2, y2],
                                        "object": activity_label,
                                        "object_bbox": None,
                                        "holding": False,
                                        "evidence": {"rule": emitted.get("evidence_rule")},
                                        "track_id": track_id,
                                    })
                            else:
                                if debug_info and debug_info.get("state") == "sleep":
                                    per_frame_activities.append({
                                        "person_id": pid,
                                        "person_bbox": [x1, y1, x2, y2],
                                        "object": "sleep",
                                        "object_bbox": None,
                                        "holding": False,
                                        "evidence": {"rule": "state_hold"},
                                        "track_id": track_id,
                                    })
                        else:
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
                                    "event_start_ts": sev.get("start_ts"),
                                    "event_end_ts": sev.get("end_ts"),
                                })

                        if self.save_debug_overlays:
                            lines = []
                            lines.append(f"EAR: {ear:.3f}" if ear is not None else "EAR: None")
                            lines.append(f"open_prob: {open_prob:.2f}" if open_prob is not None else "open_prob: None")
                            lines.append(f"head_pitch_deg: {head_down_angle:.1f}" if head_down_angle is not None else "head_pitch_deg: None")
                            tag_base = f"r{start_frame}_frame_{sampled_idx:06d}_{ts:.2f}s"
                            try:
                                save_debug_overlay(frame_bgr, lines, tag=tag_base, out_dir=os.path.join(os.path.dirname(video_path), "output"))
                            except Exception:
                                pass

                        hand_boxes_frame = []
                        for hb in hand_boxes_crop:
                            hx1, hy1, hx2, hy2 = hb
                            hand_boxes_frame.append([hx1 + x1, hy1 + y1, hx2 + x1, hy2 + y1])
                        bag_boxes_frame = []
                        pw = max(1.0, float(x2 - x1)); ph = max(1.0, float(y2 - y1))
                        person_area = pw * ph
                        torso_y1 = y1 + int(0.20 * ph)
                        torso_y2 = y1 + int(min(1.0, 0.20 + self.bag_torso_band_frac) * ph)
                        for (cid, score, ob) in objects:
                            if cid not in (24, 26, 28):
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
                            ib_y1 = max(by1, torso_y1)
                            ib_y2 = min(by2, torso_y2)
                            if (ib_y2 - ib_y1) <= 0:
                                continue
                            bag_boxes_frame.append([bx1, by1, bx2, by2])

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
                            label = str(aev.get("activity", "")).lower()
                            evidence_rule = str(aev.get("evidence_rule", ""))

                            allow_emit = True
                            evidence = {"rule": evidence_rule}

                            if label == "writing" and bool(self.write_require_surface):
                                ph = max(1.0, float(y2 - y1))
                                lap_y1 = y1 + int((1.0 - float(self.write_lap_band_frac)) * ph)
                                lap_rect = (float(x1), float(lap_y1), float(x2), float(y2))

                                has_book_surface = any(
                                    (cid == 73) and (iou(pb, ob) > 0.0) and (iou(lap_rect, ob) > 0.05)
                                    for (cid, score, ob) in objects
                                )

                                has_text_surface = False
                                if (not has_book_surface) and (ocr is not None):
                                    try:
                                        lap_crop = frame_bgr[lap_y1:y2, x1:x2]
                                        txt = (ocr.extract_text(lap_crop) or "").strip()
                                        has_text_surface = (len(txt) >= int(self.write_ocr_min_chars))
                                    except Exception:
                                        has_text_surface = False

                                allow_emit = bool(has_book_surface or has_text_surface)
                                if allow_emit:
                                    evidence = {"rule": "writing_surface_present_book" if has_book_surface else "writing_surface_present_ocr"}
                                else:
                                    evidence = {"rule": "writing_suppressed_no_surface"}

                            if allow_emit:
                                per_frame_activities.append({
                                    "person_id": pid,
                                    "person_bbox": [x1, y1, x2, y2],
                                    "object": label,
                                    "object_bbox": None,
                                    "holding": False,
                                    "evidence": evidence,
                                    "track_id": track_id,
                                })

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

                    filtered_detections = persons + objects
                    tag_base = f"r{start_frame}_frame_{sampled_idx:06d}_{ts:.2f}s"
                    result_for_annot = {
                        "detections": [
                            {"bbox": b, "conf": s, "name": "person" if c == 0 else "object"}
                            for (c, s, b) in filtered_detections
                        ],
                        "activities": per_frame_activities,
                    }
                    # Write overlays/images only when debugging or when activities are present
                    if self.save_debug_overlays or per_frame_activities:
                        try:
                            annotate_and_save(frame_bgr, result_for_annot, tag=tag_base, out_dir=os.path.join(os.path.dirname(video_path), "output"))
                        except Exception:
                            pass

                    # Use shared mapping helper
                    def _map_activity(label: str) -> Tuple[int, str]:
                        return self._map_activity_label(label)

                    for act in per_frame_activities:
                        try:
                            ev_tmp = act.get("evidence") or {}
                            if ev_tmp.get("rule") == "state_hold":
                                continue
                        except Exception:
                            pass
                        obj = str(act.get("object", "")).lower()
                        activity_type, des = _map_activity(obj)
                        if obj in ("more than two people", "more_than_two_people", "group"):
                            try:
                                ev = act.get("evidence") or {}
                                cnt = ev.get("count")
                                if isinstance(cnt, int) and cnt >= 0:
                                    des = f"{cnt} people detected"
                            except Exception:
                                pass
                        clip_filename = None
                        try:
                            out_dir = os.path.join(os.path.dirname(video_path), "output")
                            clip_filename = extract_clip(video_path, float(ts), out_dir, tag_base, duration_s=4.0)
                        except Exception:
                            clip_filename = None

                        _ev_start = act.get("event_start_ts")
                        _ev_end = act.get("event_end_ts")
                        _start_ts = float(_ev_start) if _ev_start is not None else float(ts)
                        _end_ts = float(_ev_end) if _ev_end is not None else (float(_start_ts) + 4.0)

                        event = ActivityEvent(
                            tripId=self.trip_id,
                            activityType=activity_type,
                            des=des,
                            objectType=obj,
                            fileUrl=video_path,
                            fileDuration=file_duration,
                            activityStartTime=f"{_start_ts:.2f}",
                            activityEndTime=f"{_end_ts:.2f}",
                            crewName=self.crew_name,
                            crewId=self.crew_id,
                            crewRole=self.crew_role,
                            date=date_str,
                            time=time_str,
                            filename=filename,
                            peopleCount=len(persons),
                            evidence=act.get("evidence"),
                            activityImage=f"{tag_base}_activity.jpg",
                            activityClip=clip_filename,
                        )
                        events.append(event)

                processed += 1
                sampled_idx += 1
                if callable(progress_cb):
                    try:
                        progress_cb({"processed": processed, "total": expected_total})
                    except Exception:
                        pass

        try:
            batch_frames: list = []
            batch_meta: list = []  # (frame_idx, ts)
            for frame_idx in range(first_sample, end_frame_i, step):
                # Respect max_frames early stop
                if self.max_frames and processed >= self.max_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                ts = frame_idx / max(1e-6, native_fps)
                batch_frames.append(frame_bgr)
                batch_meta.append((frame_idx, ts))
                if len(batch_frames) >= max(1, int(yolo_bs)):
                    _process_batch(batch_frames, batch_meta)
                    batch_frames.clear()
                    batch_meta.clear()
            # Flush remainder
            if batch_frames:
                _process_batch(batch_frames, batch_meta)
        finally:
            cap.release()

        pending = sleep_tracker.finalize()
        logger.info(f"[Pipeline:Range] Processed {processed} sampled frames from {filename} in range [{start_frame}, {end_frame})")
        if callable(progress_cb):
            try:
                progress_cb({"processed": processed, "total": expected_total, "done": True})
            except Exception:
                pass
        return events

