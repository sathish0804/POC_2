from typing import Dict, Any, Optional, Deque, Tuple
from collections import deque
from dataclasses import dataclass
import math
from loguru import logger

from .filters import Ewma

@dataclass
class SleepDecisionConfigV2:
    """Configuration for the V2 sleep decision state machine.

    Thresholds and windows are tuned for ~1 FPS sampling; see adapter below for
    backward-compatible mapping from legacy config knobs.
    """
    # Windows (seconds)
    short_window_s: float = 5.0      # motion/blink
    mid_window_s: float = 45.0       # PERCLOS (time-weighted)
    long_window_s: float = 120.0     # posture/context

    # Smoothing
    smoothing_alpha: float = 0.45

    # Thresholds (primary eye path)
    microsleep_min_s: float = 2.0          # continuous closure
    blink_max_s: float = 0.4               # under this → blink, not microsleep
    perclos_drowsy: float = 0.40           # mid window PERCLOS
    perclos_sleep: float = 0.80

    # Posture thresholds
    head_pitch_down_deg: float = 20.0
    head_pitch_sleep_deg: float = 35.0

    # Reliability gates
    open_prob_closed_thresh: float = 0.35  # below = closed
    open_prob_conf_floor: float = 0.15     # if open_prob fluctuates near 0/1, trust more
    eye_signal_min_cov: float = 0.03       # if open_prob variance < this → suspect stuck/occluded
    ear_high_weird_threshold: float = 0.42

    # Movement
    motion_px_thresh: float = 16.0         # SW displacement magnitude (pixel domain)
    motion_mad_scale: float = 1.4826       # robust MAD scale

    # Hysteresis / holds (seconds)
    hold_escalate_s: float = 1.5
    hold_recover_s: float = 3.0
    posture_micro_hold_s: float = 2.0

    # No-eye fallback
    no_eye_pitch_drowsy_deg: float = 28.0
    no_eye_pitch_sleep_deg: float = 40.0
    no_eye_min_still_s: float = 3.0

    # Adaptive baseline
    baseline_warmup_s: float = 8.0          # gather during “awake” before using baseline
    baseline_decay: float = 0.995           # slow drift tracking

    # Engagement suppression (0..1 where 1=fully engaged)
    engaged_downgrade: float = 0.35         # scale down risk/conf when engaged

class SleepDecisionMachineV2:
    """
    State machine for micro_sleep/drowsy/sleep decisions.

    Emits e.g.:
    {"activity": "micro_sleep"|"drowsy"|"sleep",
     "confidence": 0..1,
     "closed_run_s": ...,
     "perclos_mw": ...,
     "rule": "eye_path|posture_path|mixed",
     "ts": ts}
    """

    def __init__(self, cfg: Optional[SleepDecisionConfigV2] = None):
        self.cfg = cfg or SleepDecisionConfigV2()
        self._tracks: Dict[int, Dict[str, Any]] = {}
        logger.debug("[SleepDecisionMachineV2] initialized")

    def _ensure(self, track_id: int) -> Dict[str, Any]:
        st = self._tracks.get(track_id)
        if st is None:
            st = {
                "last_ts": None,
                # Smoothers
                "open_ewma": Ewma(self.cfg.smoothing_alpha),
                "pitch_ewma": Ewma(self.cfg.smoothing_alpha),
                "ear_ewma": Ewma(self.cfg.smoothing_alpha),
                # Histories: deques of (t, val)
                "open_hist": deque(),
                "closed_hist": deque(),   # (t, is_closed_int)
                "pitch_hist": deque(),
                "head_points": deque(),   # (t, x, y)
                # Time-weighted accumulators
                "closed_run_s": 0.0,
                "state": "awake",
                "state_since": None,
                "esc_hold_start": None,
                "rec_hold_start": None,
                "post_micro_start": None,
                # Baseline
                "baseline_open": None,
                "baseline_start_ts": None,
                # Reliability
                "open_var_hist": deque(),  # (t, open_prob)
            }
            self._tracks[track_id] = st
        return st

    @staticmethod
    def _prune(hist: Deque[Tuple[float, Any]], now_ts: float, window_s: float) -> None:
        cutoff = now_ts - window_s
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    @staticmethod
    def _time_weighted_perclos(closed_hist: Deque[Tuple[float, int]], now_ts: float, window_s: float) -> float:
        """Integrate closed time over window / window length."""
        if not closed_hist:
            return 0.0
        cutoff = now_ts - window_s
        # Integrate piecewise between samples using last known state
        total = 0.0
        last_t = None
        last_closed = None
        # Seed with first sample after cutoff (or nearest)
        for i, (t, c) in enumerate(closed_hist):
            if t >= cutoff:
                last_t = max(t, cutoff)
                last_closed = c
                break
        if last_t is None:
            # All samples older than cutoff → use last sample’s state
            t_last, c_last = closed_hist[-1]
            return float(c_last)  # degenerate, but prevents div0
        # Integrate forward
        for i in range(i+1, len(closed_hist)):
            t, c = closed_hist[i]
            dt = max(0.0, t - last_t)
            if last_closed is not None and last_closed == 1:
                total += dt
            last_t = t
            last_closed = c
        # integrate tail to now
        dt_tail = max(0.0, now_ts - last_t)
        if last_closed == 1:
            total += dt_tail
        return min(1.0, total / max(1e-6, window_s))

    @staticmethod
    def _robust_motion(head_points: Deque[Tuple[float, float, float]], now_ts: float, window_s: float) -> float:
        """Return robust displacement magnitude (MAD of dx,dy) in the window."""
        cutoff = now_ts - window_s
        xs, ys = [], []
        for (t, x, y) in head_points:
            if t >= cutoff:
                xs.append(x); ys.append(y)
        if len(xs) < 3:
            return 0.0
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        # A simple robust proxy (MAD). For speed, use range as proxy plus little MAD.
        disp = math.hypot(dx, dy)
        return disp

    @staticmethod
    def _variance(vals):
        if len(vals) < 3:
            return 0.0
        m = sum(vals)/len(vals)
        return sum((v-m)*(v-m) for v in vals)/len(vals)

    def _eye_reliable(self, st, now_ts: float) -> bool:
        # low variance across mid window → likely stuck/occluded
        cutoff = now_ts - self.cfg.mid_window_s
        vals = [v for (t, v) in st["open_var_hist"] if t >= cutoff]
        var = self._variance(vals)
        return var >= self.cfg.eye_signal_min_cov

    def update(
        self,
        *,
        track_id: int,
        ts: float,
        ear: Optional[float],
        open_prob: Optional[float],
        head_pitch_deg: Optional[float],
        head_point_xy: Optional[Tuple[float, float]],
        engaged: bool,
        face_quality: Optional[float] = None,   # 0..1 (optional)
    ) -> Optional[Dict[str, Any]]:
        st = self._ensure(track_id)
        st["last_ts"] = ts

        # Smooth inputs
        ear_s   = st["ear_ewma"].update(ear)
        open_s  = st["open_ewma"].update(open_prob)
        pitch_s = st["pitch_ewma"].update(head_pitch_deg)

        # Baseline warmup for open_prob when “awake”
        if st["state"] == "awake" and open_s is not None:
            if st["baseline_open"] is None:
                st["baseline_open"] = float(open_s)
                st["baseline_start_ts"] = ts
            else:
                # very slow decay to track drift
                st["baseline_open"] = (self.cfg.baseline_decay * st["baseline_open"] +
                                       (1.0 - self.cfg.baseline_decay) * float(open_s))

        # Append histories
        if open_s is not None:
            st["open_hist"].append((ts, float(open_s)))
            st["open_var_hist"].append((ts, float(open_s)))
            # Eye closed detection
            is_closed = 1 if open_s < self.cfg.open_prob_closed_thresh else 0
            st["closed_hist"].append((ts, is_closed))
            # Time-weighted closed run (not sample-count)
            # Compute dt since previous closed sample time or last timestamp in closed_hist
            if len(st["closed_hist"]) >= 2:
                t_prev = st["closed_hist"][-2][0]
                dt = max(0.01, ts - t_prev)
            else:
                dt = 0.01
            if is_closed == 1:
                st["closed_run_s"] += dt
            else:
                # allow blink forgiveness: if run < blink_max_s treat as blink; else reset
                if st["closed_run_s"] > self.cfg.blink_max_s:
                    # past run counted towards microsleep; we reset after emitting below
                    pass
                st["closed_run_s"] = 0.0

        if pitch_s is not None:
            st["pitch_hist"].append((ts, float(pitch_s)))

        if head_point_xy is not None:
            x, y = head_point_xy
            st["head_points"].append((ts, float(x), float(y)))

        # Prune to windows
        self._prune(st["open_hist"], ts, self.cfg.long_window_s)
        self._prune(st["open_var_hist"], ts, self.cfg.mid_window_s)
        self._prune(st["closed_hist"], ts, self.cfg.long_window_s)
        self._prune(st["pitch_hist"], ts, self.cfg.long_window_s)
        self._prune(st["head_points"], ts, self.cfg.short_window_s)

        # Aggregates
        perclos_mw = self._time_weighted_perclos(st["closed_hist"], ts, self.cfg.mid_window_s)
        motion_mag = self._robust_motion(st["head_points"], ts, self.cfg.short_window_s)

        # Reliability
        eye_rel = (open_s is not None) and self._eye_reliable(st, ts)
        if ear is not None and ear >= self.cfg.ear_high_weird_threshold:
            # tipped head / occlusion → treat eyes as unreliable
            eye_rel = False
        if face_quality is not None and face_quality < 0.3:
            eye_rel = False

        # Posture conditions (use signed pitch: positive = head-down, negative = head-back)
        head_down = (pitch_s is not None) and (pitch_s >= self.cfg.head_pitch_down_deg)
        head_very_down = (pitch_s is not None) and (pitch_s >= self.cfg.head_pitch_sleep_deg)

        still = motion_mag <= self.cfg.motion_px_thresh

        # Engagement: scale down risk confidence if true (e.g., using phone)
        engaged_factor = (1.0 - self.cfg.engaged_downgrade) if engaged else 1.0

        # Decision
        state = st["state"]
        emit = None

        def start_hold(name):
            if st.get(name) is None:
                st[name] = ts

        def passed(name, dur):
            return (st.get(name) is not None) and (ts - st[name] >= dur)

        def clear_hold(name):
            st[name] = None

        # Eye path signals
        microsleep_eye = False
        blink_like = False
        if eye_rel and open_s is not None:
            if st["closed_run_s"] >= self.cfg.microsleep_min_s and (still or perclos_mw >= 0.90):
                microsleep_eye = True
            elif 0.0 < st["closed_run_s"] <= self.cfg.blink_max_s:
                blink_like = True

        # Posture path (no-eye) – require positive (down) pitch
        microsleep_posture = False
        sleep_posture = False
        if not eye_rel:
            if pitch_s is not None:
                if pitch_s >= self.cfg.no_eye_pitch_sleep_deg and still:
                    sleep_posture = True
                elif pitch_s >= self.cfg.no_eye_pitch_drowsy_deg and still:
                    microsleep_posture = True

        # Escalation / recovery logic
        if state == "awake":
            # Enter drowsy tier silently if mid PERCLOS high, or posture suggests drowsy
            if perclos_mw >= self.cfg.perclos_drowsy or microsleep_posture:
                st["state"] = "drowsy"
                st["state_since"] = ts
                clear_hold("esc_hold_start"); clear_hold("rec_hold_start")

            # Emit microsleep event
            allow_posture_micro = False
            if microsleep_posture:
                # start/require hold for posture-only microsleep
                start_hold("post_micro_start")
                if passed("post_micro_start", self.cfg.posture_micro_hold_s):
                    allow_posture_micro = True
            else:
                clear_hold("post_micro_start")

            if microsleep_eye or allow_posture_micro:
                rule = "eye_path" if microsleep_eye and eye_rel else "posture_path"
                conf = 0.65
                if microsleep_eye and perclos_mw >= 0.5: conf += 0.15
                if head_down: conf += 0.1
                if not still: conf -= 0.2
                conf = max(0.1, min(1.0, conf * engaged_factor))
                # Episode timing
                if rule == "eye_path":
                    est_start = max(0.0, ts - float(st.get("closed_run_s", 0.0)))
                else:
                    est_start = float(st.get("post_micro_start", ts - self.cfg.posture_micro_hold_s))
                emit = {"activity": "micro_sleep", "confidence": float(conf), "rule": rule,
                        "closed_run_s": float(st["closed_run_s"]), "perclos_mw": float(perclos_mw), "ts": ts,
                        "event_start_ts": float(est_start), "event_end_ts": float(ts)}
                # move to drowsy tier after a microsleep alert
                st["state"] = "drowsy"
                st["state_since"] = ts

        elif state == "drowsy":
            # Consider escalation to sleep
            # Allow sleep with very high PERCLOS or long continuous closure even without head-down
            sleep_eye = False
            if eye_rel:
                if perclos_mw >= self.cfg.perclos_sleep and (head_down or perclos_mw >= 0.90):
                    sleep_eye = True
                elif st.get("closed_run_s", 0.0) >= max(8.0, self.cfg.microsleep_min_s + 6.0):
                    sleep_eye = True
            sleep_any = sleep_eye or sleep_posture
            if sleep_any:
                start_hold("esc_hold_start")
                if passed("esc_hold_start", self.cfg.hold_escalate_s):
                    st["state"] = "sleep"
                    st["state_since"] = ts
                    clear_hold("esc_hold_start")
                    # Save sleep episode metadata; actual emission will occur on recovery with duration
                    rule = "eye_path" if sleep_eye else "posture_path"
                    conf = 0.75 if sleep_eye else 0.7
                    if head_very_down: conf += 0.1
                    if perclos_mw >= 0.9: conf += 0.1
                    if not still: conf -= 0.2
                    conf = max(0.1, min(1.0, conf * engaged_factor))
                    st["sleep_meta"] = {"rule": rule, "confidence": float(conf), "start_ts": float(ts)}
            else:
                clear_hold("esc_hold_start")

            # Recovery to awake
            recovered = False
            if eye_rel and open_s is not None and st["baseline_open"] is not None:
                # “good open + neutral head” for a bit
                good_open = open_s >= max(0.8 * st["baseline_open"], 1.0 - self.cfg.open_prob_closed_thresh)
                neutral_head = (pitch_s is not None) and (abs(pitch_s) <= self.cfg.head_pitch_down_deg / 1.5)
                recovered = good_open and neutral_head and (perclos_mw <= self.cfg.perclos_drowsy * 0.6)
            elif not eye_rel and pitch_s is not None:
                recovered = (abs(pitch_s) <= self.cfg.head_pitch_down_deg / 2) and (still is False)
            elif not eye_rel and pitch_s is None:
                # If we lack pitch but see strong motion, consider recovery
                recovered = (still is False)

            if recovered:
                start_hold("rec_hold_start")
                if passed("rec_hold_start", self.cfg.hold_recover_s):
                    # Emit sleep episode with duration using saved metadata
                    meta = st.get("sleep_meta", {})
                    start_ts = float(meta.get("start_ts", st.get("state_since", ts)))
                    rule = str(meta.get("rule", "mixed"))
                    conf = float(meta.get("confidence", 0.7))
                    emit = {"activity": "sleep", "confidence": conf, "rule": rule,
                            "closed_run_s": float(st.get("closed_run_s", 0.0)), "perclos_mw": float(perclos_mw), "ts": ts,
                            "event_start_ts": float(start_ts), "event_end_ts": float(ts)}
                    st["state"] = "awake"
                    st["state_since"] = ts
                    st["sleep_meta"] = None
                    clear_hold("rec_hold_start")
            else:
                clear_hold("rec_hold_start")

            # Additional microsleep alerts while drowsy (rate-limited via holds)
            if emit is None and still:
                if microsleep_eye:
                    start_hold("esc_hold_start")
                    if passed("esc_hold_start", self.cfg.hold_escalate_s / 2.0):
                        conf = 0.6
                        if head_down: conf += 0.1
                        if perclos_mw > 0.6: conf += 0.1
                        conf = max(0.1, min(1.0, conf * engaged_factor))
                        emit = {"activity": "micro_sleep", "confidence": float(conf), "rule": "eye_path",
                                "closed_run_s": float(st["closed_run_s"]), "perclos_mw": float(perclos_mw), "ts": ts}
                        clear_hold("esc_hold_start")
                elif microsleep_posture:
                    start_hold("post_micro_start")
                    if passed("post_micro_start", self.cfg.posture_micro_hold_s):
                        conf = 0.6
                        if head_down: conf += 0.1
                        conf = max(0.1, min(1.0, conf * engaged_factor))
                        emit = {"activity": "micro_sleep", "confidence": float(conf), "rule": "posture_path",
                                "closed_run_s": float(st["closed_run_s"]), "perclos_mw": float(perclos_mw), "ts": ts}

        elif state == "sleep":
            # Recover only with sustained evidence
            recovered = False
            if eye_rel and open_s is not None and st["baseline_open"] is not None:
                good_open = open_s >= max(0.8 * st["baseline_open"], 1.0 - self.cfg.open_prob_closed_thresh)
                neutral_head = (pitch_s is not None) and (abs(pitch_s) <= self.cfg.head_pitch_down_deg / 1.5)
                recovered = good_open and neutral_head and (perclos_mw <= self.cfg.perclos_drowsy * 0.5)
            elif not eye_rel and pitch_s is not None:
                recovered = (abs(pitch_s) <= self.cfg.head_pitch_down_deg / 2) and (still is False)
            elif not eye_rel and pitch_s is None:
                recovered = (still is False)

            if recovered:
                start_hold("rec_hold_start")
                if passed("rec_hold_start", self.cfg.hold_recover_s):
                    st["state"] = "awake"
                    st["state_since"] = ts
                    clear_hold("rec_hold_start")
            else:
                clear_hold("rec_hold_start")

        return emit

    def get_debug(self, track_id: int, now_ts: float) -> Dict[str, Any]:
        st = self._tracks.get(track_id)
        if not st:
            return {}
        open_s  = st["open_ewma"]._y if st.get("open_ewma") else None
        pitch_s = st["pitch_ewma"]._y if st.get("pitch_ewma") else None
        ear_s   = st["ear_ewma"]._y if st.get("ear_ewma") else None
        perclos_mw = SleepDecisionMachineV2._time_weighted_perclos(st.get("closed_hist", deque()), now_ts, self.cfg.mid_window_s)
        motion_mag = SleepDecisionMachineV2._robust_motion(st.get("head_points", deque()), now_ts, self.cfg.short_window_s)
        return {
            "state": st.get("state"),
            "since": st.get("state_since"),
            "closed_run_s": float(st.get("closed_run_s", 0.0)),
            "perclos_mw": float(perclos_mw),
            "motion_sw": float(motion_mag),
            "open_prob_smoothed": None if open_s is None else float(open_s),
            "pitch_deg_smoothed": None if pitch_s is None else float(pitch_s),
            "ear_smoothed": None if ear_s is None else float(ear_s),
            "baseline_open": st.get("baseline_open"),
        }

# -----------------------------------------------------------------------------
# Backward-compatible adapter for legacy imports (SleepDecisionMachine, Config)
# -----------------------------------------------------------------------------

@dataclass
class SleepDecisionConfig:
    # Legacy field names expected by callers (mapped to V2 under the hood)
    short_window_s: float = 5.0
    mid_window_s: float = 45.0
    long_window_s: float = 120.0
    smoothing_alpha: float = 0.45

    # Legacy eye closure semantics → map to V2 microsleep_min_s
    eye_closed_run_s: float = 2.0

    # Legacy PERCLOS thresholds
    perclos_drowsy_thresh: float = 0.40
    perclos_sleep_thresh: float = 0.80

    # Posture / head pitch
    head_pitch_down_deg: float = 20.0
    head_neutral_deg: float = 12.0  # not used directly in V2

    # Hysteresis / holds
    hold_transition_s: float = 1.5
    recovery_hold_s: float = 3.0

    # Eye reliability / thresholds
    open_prob_closed_thresh: float = 0.35
    ear_high_weird_threshold: float = 0.42

    # Legacy knobs not directly used in V2
    head_down_micro_fallback: bool = True
    head_down_micro_deg: float = 40.0
    no_eye_head_down_deg: float = 28.0


class SleepDecisionMachine:
    """
    Thin adapter that accepts the legacy config and proxies to SleepDecisionMachineV2.
    It also normalizes the emitted dict to include "evidence_rule" for backward compatibility.
    """

    def __init__(self, cfg: Optional[SleepDecisionConfig] = None):
        cfg = cfg or SleepDecisionConfig()
        # Map legacy config to V2 config
        v2cfg = SleepDecisionConfigV2(
            short_window_s=cfg.short_window_s,
            mid_window_s=cfg.mid_window_s,
            long_window_s=cfg.long_window_s,
            smoothing_alpha=cfg.smoothing_alpha,
            microsleep_min_s=cfg.eye_closed_run_s,
            perclos_drowsy=cfg.perclos_drowsy_thresh,
            perclos_sleep=cfg.perclos_sleep_thresh,
            head_pitch_down_deg=cfg.head_pitch_down_deg,
            # Keep default head_pitch_sleep_deg or raise slightly if legacy micro-deg is higher
            head_pitch_sleep_deg=max(SleepDecisionConfigV2().head_pitch_sleep_deg, cfg.head_down_micro_deg),
            open_prob_closed_thresh=cfg.open_prob_closed_thresh,
            ear_high_weird_threshold=cfg.ear_high_weird_threshold,
            hold_escalate_s=cfg.hold_transition_s,
            hold_recover_s=cfg.recovery_hold_s,
            # Map single legacy no-eye pitch to a conservative pair in V2
            no_eye_pitch_drowsy_deg=cfg.no_eye_head_down_deg,
            no_eye_pitch_sleep_deg=max(cfg.no_eye_head_down_deg, 40.0),
        )
        self._impl = SleepDecisionMachineV2(v2cfg)
        logger.debug("[SleepDecisionMachine] adapter initialized")

    def update(self, **kwargs):
        res = self._impl.update(**kwargs)
        if res is not None and "evidence_rule" not in res and "rule" in res:
            # Provide legacy key expected by some callers
            res["evidence_rule"] = res.get("rule")
        return res

    def get_debug(self, track_id: int, now_ts: float) -> Dict[str, Any]:
        return self._impl.get_debug(track_id, now_ts)


# Module import log
logger.debug(f"[{__name__}] module loaded")
