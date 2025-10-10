from typing import Dict, Any, Optional, Deque, Tuple
from collections import deque
from dataclasses import dataclass

from .filters import Ewma


@dataclass
class SleepDecisionConfig:
    # Windows (seconds)
    short_window_s: float = 5.0   # SW: 3–5 s
    mid_window_s: float = 45.0    # MW: 30–60 s
    long_window_s: float = 120.0  # LW: 120 s

    # Smoothing
    smoothing_alpha: float = 0.4

    # Thresholds
    eye_closed_run_s: float = 2.0           # microsleep continuous closure seconds
    perclos_drowsy_thresh: float = 0.40     # 0.35–0.45
    perclos_sleep_thresh: float = 0.80      # 0.80
    head_pitch_down_deg: float = 20.0       # head down vs neutral
    head_neutral_deg: float = 15.0          # for recovery

    # Hysteresis/holds (seconds)
    hold_transition_s: float = 2.0
    recovery_hold_s: float = 3.0

    # Eye-open probability threshold (maps to “closed” when below)
    open_prob_closed_thresh: float = 0.35
    # Fallbacks
    head_down_micro_fallback: bool = True   # allow microsleep fallback on strong head-down + stillness
    head_down_micro_deg: float = 50.0       # degrees for fallback
    ear_high_weird_threshold: float = 0.42  # EAR above this likely unreliable (tipped head/occlusion)


class SleepDecisionMachine:
    """Stateful decision logic per track with sliding windows + hysteresis.

    Emits events: {"activity": "micro_sleep"|"drowsy"|"sleep", ...}
    """

    def __init__(self, config: Optional[SleepDecisionConfig] = None):
        self.cfg = config or SleepDecisionConfig()
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def _ensure(self, track_id: int) -> Dict[str, Any]:
        st = self._tracks.get(track_id)
        if st is None:
            st = {
                "last_ts": None,
                "ear_ewma": Ewma(self.cfg.smoothing_alpha),
                "open_ewma": Ewma(self.cfg.smoothing_alpha),
                "pitch_ewma": Ewma(self.cfg.smoothing_alpha),
                # Rolling windows as deques of (ts, value)
                "open_hist": deque(),   # (ts, open_prob in [0,1])
                "closed_hist": deque(), # (ts, is_closed bool)
                "pitch_hist": deque(),  # (ts, pitch_deg)
                # Timers and state
                "closed_run_s": 0.0,
                "state": "awake",  # awake | drowsy | sleep
                "state_since": None,
                "pending_event": None,
                # For motion estimation: track head point (e.g., nose)
                "head_points": deque(),  # (ts, x, y)
            }
            self._tracks[track_id] = st
        return st

    @staticmethod
    def _prune_window(hist: Deque[Tuple[float, Any]], now_ts: float, window_s: float) -> None:
        cutoff = now_ts - window_s
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    @staticmethod
    def _perclos(hist_closed: Deque[Tuple[float, int]], now_ts: float, window_s: float) -> float:
        # Approximate PERCLOS by proportion of samples flagged closed within window
        if not hist_closed:
            return 0.0
        # Count samples; assumes roughly uniform sampling
        n = 0
        c = 0
        cutoff = now_ts - window_s
        for (t, is_closed) in hist_closed:
            if t >= cutoff:
                n += 1
                c += 1 if is_closed else 0
        if n == 0:
            return 0.0
        return c / float(max(1, n))

    @staticmethod
    def _low_head_motion(head_points: Deque[Tuple[float, float, float]], now_ts: float, window_s: float, px_thresh: float = 12.0) -> bool:
        # Simple displacement-based motion proxy over SW
        cutoff = now_ts - window_s
        xs = []
        ys = []
        for (t, x, y) in head_points:
            if t >= cutoff:
                xs.append(x)
                ys.append(y)
        if len(xs) < 2:
            return True
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        return (dx * dx + dy * dy) ** 0.5 <= px_thresh

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
    ) -> Optional[Dict[str, Any]]:
        st = self._ensure(track_id)
        st["last_ts"] = ts

        # Smooth features
        ear_s = st["ear_ewma"].update(ear)
        open_s = st["open_ewma"].update(open_prob)
        pitch_s = st["pitch_ewma"].update(head_pitch_deg)

        # Booleans and histories
        is_closed = None
        if open_s is not None:
            is_closed = 1 if open_s < self.cfg.open_prob_closed_thresh else 0
            st["open_hist"].append((ts, float(open_s)))
            st["closed_hist"].append((ts, is_closed))
            # Update continuous closure length
            if is_closed == 1:
                # Estimate dt; fall back to 1.0s if unavailable (low FPS edge)
                dt = 1.0
                if len(st["closed_hist"]) >= 2:
                    t_prev = st["closed_hist"][-2][0]
                    dt = max(0.01, ts - t_prev)
                st["closed_run_s"] += dt
            else:
                st["closed_run_s"] = 0.0

        if pitch_s is not None:
            st["pitch_hist"].append((ts, float(pitch_s)))

        if head_point_xy is not None:
            x, y = head_point_xy
            st["head_points"].append((ts, float(x), float(y)))

        # Prune histories to relevant windows
        self._prune_window(st["open_hist"], ts, self.cfg.long_window_s)
        self._prune_window(st["closed_hist"], ts, self.cfg.long_window_s)
        self._prune_window(st["pitch_hist"], ts, self.cfg.long_window_s)
        self._prune_window(st["head_points"], ts, self.cfg.short_window_s)

        # Domain suppressions
        if engaged and is_closed is not None and is_closed == 1:
            # If actively engaged (e.g., phone in hand), suppress immediate microsleep
            pass

        # Compute aggregates
        perclos_mw = self._perclos(st["closed_hist"], ts, self.cfg.mid_window_s)
        low_motion_sw = self._low_head_motion(st["head_points"], ts, self.cfg.short_window_s)
        head_down = (pitch_s is not None) and (pitch_s >= self.cfg.head_pitch_down_deg)

        # Decision with hysteresis
        state = st["state"]
        emitted: Optional[Dict[str, Any]] = None

        # Recovery condition
        recovered = False
        if open_s is not None and pitch_s is not None:
            if open_s >= (1.0 - self.cfg.open_prob_closed_thresh) and abs(pitch_s) <= self.cfg.head_neutral_deg:
                recovered = True

        if state == "awake":
            # Head-down fallback for cases where EAR is unreliable but posture indicates microsleep
            weird_ear = (ear is not None) and (ear >= self.cfg.ear_high_weird_threshold)
            if self.cfg.head_down_micro_fallback and weird_ear and head_down and low_motion_sw:
                st["state"] = "drowsy"
                st["state_since"] = ts
                emitted = {"activity": "micro_sleep", "evidence_rule": "head_down_still_fallback"}
            # Standard rules
            # Microsleep: immediate alert after continuous closure + low motion in SW
            if is_closed == 1 and st["closed_run_s"] >= self.cfg.eye_closed_run_s and low_motion_sw:
                st["state"] = "drowsy"  # enter drowsy tier after microsleep alert
                st["state_since"] = ts
                emitted = {"activity": "micro_sleep", "evidence_rule": "eye_closed_run_and_low_motion"}
            # Drowsy tier suppressed: keep internal state for escalation, but DO NOT emit a drowsy event
            elif perclos_mw >= self.cfg.perclos_drowsy_thresh:
                st["state"] = "drowsy"
                st["state_since"] = ts
                # no emission

        elif state == "drowsy":
            # Escalate to sleep: very high PERCLOS + stillness + head-down
            if perclos_mw >= self.cfg.perclos_sleep_thresh and low_motion_sw and head_down:
                # Require short hold to prevent flapping
                if st.get("hold_start") is None:
                    st["hold_start"] = ts
                if ts - st["hold_start"] >= self.cfg.hold_transition_s:
                    st["state"] = "sleep"
                    st["state_since"] = ts
                    st["hold_start"] = None
                    emitted = {"activity": "sleep", "evidence_rule": "perclos_high_stillness_head_down"}
            else:
                st["hold_start"] = None

            # Recovery from drowsy back to awake
            if recovered:
                if st.get("recover_start") is None:
                    st["recover_start"] = ts
                if ts - st["recover_start"] >= self.cfg.recovery_hold_s:
                    st["state"] = "awake"
                    st["state_since"] = ts
                    st["recover_start"] = None
            else:
                st["recover_start"] = None

        elif state == "sleep":
            # Remain until recovery condition sustained
            if recovered:
                if st.get("recover_start") is None:
                    st["recover_start"] = ts
                if ts - st["recover_start"] >= self.cfg.recovery_hold_s:
                    st["state"] = "awake"
                    st["state_since"] = ts
                    st["recover_start"] = None
            else:
                st["recover_start"] = None

        return emitted

    def get_debug(self, track_id: int, now_ts: float) -> Dict[str, Any]:
        st = self._tracks.get(track_id)
        if not st:
            return {}
        # Smoothed values
        ear_s = st.get("ear_ewma")._y if st.get("ear_ewma") else None
        open_s = st.get("open_ewma")._y if st.get("open_ewma") else None
        pitch_s = st.get("pitch_ewma")._y if st.get("pitch_ewma") else None
        # Aggregates
        perclos_mw = self._perclos(st.get("closed_hist", deque()), now_ts, self.cfg.mid_window_s)
        low_motion_sw = self._low_head_motion(st.get("head_points", deque()), now_ts, self.cfg.short_window_s)
        return {
            "state": st.get("state"),
            "closed_run_s": float(st.get("closed_run_s", 0.0)),
            "perclos_mw": float(perclos_mw),
            "low_motion_sw": bool(low_motion_sw),
            "ear_smoothed": None if ear_s is None else float(ear_s),
            "open_prob_smoothed": None if open_s is None else float(open_s),
            "pitch_deg_smoothed": None if pitch_s is None else float(pitch_s),
        }


