from typing import Dict, Any, Optional, Tuple, List
import math


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)


def compute_ear(face_landmarks: Any, frame_shape: Tuple[int, int]) -> Optional[float]:
    """Compute Eye Aspect Ratio (EAR) from MediaPipe FaceMesh landmarks.

    Uses multiple vertical pairs per eye for robustness and averages both eyes.
    Returns None when landmarks are unavailable.
    """
    if face_landmarks is None or getattr(face_landmarks, "multi_face_landmarks", None) is None:
        return None
    H, W = frame_shape
    try:
        lms = face_landmarks.multi_face_landmarks[0].landmark
    except Exception:
        return None

    # MediaPipe FaceMesh indices for eyes (refine_landmarks=True recommended)
    # Left eye corners
    L_OUT, L_IN = 33, 133
    # Left eye vertical pairs (upper, lower)
    L_PAIRS = [(159, 145), (160, 144), (158, 153)]
    # Right eye corners
    R_OUT, R_IN = 362, 263
    # Right eye vertical pairs (upper, lower)
    R_PAIRS = [(386, 374), (385, 380), (387, 373)]

    def pt(idx: int) -> Tuple[float, float]:
        return (lms[idx].x * W, lms[idx].y * H)

    try:
        l_outer = pt(L_OUT); l_inner = pt(L_IN)
        r_outer = pt(R_OUT); r_inner = pt(R_IN)
        # Aggregate vertical distances
        def vmean(pairs):
            vals: List[float] = []
            for (a, b) in pairs:
                try:
                    va = pt(a); vb = pt(b)
                    vals.append(_dist(va, vb))
                except Exception:
                    pass
            if not vals:
                return None
            # Use median-like robust average
            vals.sort()
            return sum(vals[len(vals)//2-1:len(vals)//2+1]) / 2.0 if len(vals) >= 2 else vals[0]

        l_v = vmean(L_PAIRS)
        r_v = vmean(R_PAIRS)
        if l_v is None or r_v is None:
            return None
        l_w = max(1e-6, _dist(l_outer, l_inner))
        r_w = max(1e-6, _dist(r_outer, r_inner))
        left_ear = l_v / l_w
        right_ear = r_v / r_w
        ear = (left_ear + right_ear) / 2.0
        # Clamp to sane range to avoid outliers
        return max(0.02, min(0.7, ear))
    except Exception:
        return None


def eye_open_probability(ear: Optional[float], ear_open: float = 0.28, ear_closed: float = 0.18) -> Optional[float]:
    """Map EAR to a coarse eye open probability between 0 and 1 using linear ramp.

    This is a placeholder until a dedicated NIR eye-state classifier is integrated.
    Returns None if EAR is None.
    """
    if ear is None:
        return None
    if ear <= ear_closed:
        return 0.0
    if ear >= ear_open:
        return 1.0
    # Linear interpolation
    span = max(1e-6, ear_open - ear_closed)
    return (ear - ear_closed) / span


