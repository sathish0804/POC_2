from typing import Optional, Dict, Any
import numpy as np
import mediapipe as mp
from loguru import logger


class MediaPipeService:
    """CPU MediaPipe trackers: face mesh, hands, and pose."""

    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        # Configure for CPU; default graphs run on CPU in Python
        self.face = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)
        self.pose = self.mp_pose.Pose(static_image_mode=False)
        logger.debug("[MediaPipeService] initialized")

    def process(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out['face'] = self.face.process(image_rgb)
        out['hands'] = self.hands.process(image_rgb)
        out['pose'] = self.pose.process(image_rgb)
        return out

    def close(self):
        self.face.close()
        self.hands.close()
        self.pose.close()
        logger.debug("[MediaPipeService] closed")


# Module import log
logger.debug(f"[{__name__}] module loaded")
