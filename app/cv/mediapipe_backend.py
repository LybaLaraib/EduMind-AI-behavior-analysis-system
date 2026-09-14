"""
MediaPipe Tasks API backend (0.10+) — face mesh, hands, and mesh drawing.
Downloads model files on first use into the project models/ folder.
"""

from __future__ import annotations

import logging
import os
import urllib.request

import numpy as np

from config import BASE_DIR

MODELS_DIR = os.path.join(BASE_DIR, "models")
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

logger = logging.getLogger(__name__)

# BGR colors for mesh overlay
MESH_LINE_BGR = (255, 220, 120)
MESH_POINT_BGR = (255, 255, 255)


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
        return
    urllib.request.urlretrieve(url, dest)


class _LmPoint:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class MediaPipeVisionBackend:
    """Face + hand landmark detection using MediaPipe Tasks."""

    def __init__(self) -> None:
        self._face = None
        self._hands = None
        self._available = False
        self._init()

    def _init(self) -> None:
        try:
            from mediapipe.tasks.python.core import base_options as base_options_module
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )
            from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

            self._Image = Image
            self._ImageFormat = ImageFormat

            _download(FACE_MODEL_URL, FACE_MODEL_PATH)
            _download(HAND_MODEL_URL, HAND_MODEL_PATH)

            face_opts = FaceLandmarkerOptions(
                base_options=base_options_module.BaseOptions(
                    model_asset_path=FACE_MODEL_PATH
                ),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            hand_opts = HandLandmarkerOptions(
                base_options=base_options_module.BaseOptions(
                    model_asset_path=HAND_MODEL_PATH
                ),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self._face = FaceLandmarker.create_from_options(face_opts)
            self._hands = HandLandmarker.create_from_options(hand_opts)
            self._available = True
            logger.info("MediaPipe face + hand landmarkers ready")
        except Exception as exc:
            logger.warning("MediaPipe init failed — using OpenCV fallback: %s", exc)
            self._face = None
            self._hands = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def process_face(self, rgb: np.ndarray):
        if not self._face:
            return None
        mp_image = self._Image(
            image_format=self._ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        return self._face.detect(mp_image)

    def process_hands(self, rgb: np.ndarray):
        if not self._hands:
            return None
        mp_image = self._Image(
            image_format=self._ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        return self._hands.detect(mp_image)

    @staticmethod
    def landmarks_to_adapter(landmark_list) -> list[_LmPoint]:
        return [_LmPoint(lm.x, lm.y, getattr(lm, "z", 0.0) or 0.0) for lm in landmark_list]

    def draw_face_mesh(self, bgr_frame: np.ndarray, landmark_list) -> np.ndarray:
        if not landmark_list:
            return bgr_frame
        try:
            import cv2
            from mediapipe.tasks.python.vision import drawing_utils
            from mediapipe.tasks.python.vision.face_landmarker import (
                FaceLandmarksConnections,
            )

            line_spec = drawing_utils.DrawingSpec(
                color=MESH_LINE_BGR, thickness=1, circle_radius=1
            )
            point_spec = drawing_utils.DrawingSpec(
                color=MESH_POINT_BGR, thickness=1, circle_radius=2
            )
            drawing_utils.draw_landmarks(
                bgr_frame,
                landmark_list,
                FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=point_spec,
                connection_drawing_spec=line_spec,
            )
            drawing_utils.draw_landmarks(
                bgr_frame,
                landmark_list,
                FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=line_spec,
            )
        except Exception as exc:
            logger.debug("Face mesh draw skipped: %s", exc)
        return bgr_frame

    def close(self) -> None:
        if self._face:
            self._face.close()
            self._face = None
        if self._hands:
            self._hands.close()
            self._hands = None


def hand_landmarks_list(hand_result) -> list:
    """Normalize hand result to list of landmark lists."""
    if not hand_result or not hand_result.hand_landmarks:
        return []
    return hand_result.hand_landmarks
