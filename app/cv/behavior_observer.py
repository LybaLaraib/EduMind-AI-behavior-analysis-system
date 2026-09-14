"""
OpenCV + MediaPipe observation layer.
Extracts behavioral facts — posture, eyes, hands/mobile, face presence.
Uses temporal gating so facts require sustained cues before assertion.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.cv.fact_temporal import FactTemporalGate
from app.cv.mediapipe_backend import MediaPipeVisionBackend, hand_landmarks_list
from config import (
    EAR_CLOSED_RATIO,
    EAR_CLOSED_THRESHOLD,
    EAR_SMOOTH_ALPHA,
    EYES_CLOSED_MIN_SECONDS,
    HAND_NEAR_FACE_MIN_SECONDS,
    HAND_NEAR_FACE_RATIO,
    HEAD_DOWN_MIN_SECONDS,
    HEAD_PITCH_DOWN_THRESHOLD,
    HEAD_YAW_THRESHOLD,
    LOOKING_AWAY_MIN_SECONDS,
    NO_FACE_MIN_SECONDS,
    PHONE_BELOW_FACE_RATIO,
    PHONE_SUSPECT_MIN_SECONDS,
)

POSE_SMOOTH_ALPHA = 0.35


@dataclass
class ObservationResult:
    facts: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    labels: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": self.facts,
            "metrics": self.metrics,
            "labels": self.labels,
            "message": self.message,
        }


class BehaviorObserver:
    """Detects classroom behavior cues and maps them to Prolog-ready facts."""

    def __init__(self):
        self._backend = MediaPipeVisionBackend()
        self._blink_times: list[float] = []
        self._last_ear = 1.0
        self._ear_smoothed = 0.3
        self._ear_baseline = 0.0
        self._ear_samples = 0
        self._smooth_yaw = 0.0
        self._smooth_pitch = 0.0
        self._has_smooth = False
        self._last_draw_landmarks = None

        self._gate_eyes = FactTemporalGate(EYES_CLOSED_MIN_SECONDS)
        self._gate_looking_away = FactTemporalGate(LOOKING_AWAY_MIN_SECONDS)
        self._gate_head_down = FactTemporalGate(HEAD_DOWN_MIN_SECONDS)
        self._gate_phone = FactTemporalGate(PHONE_SUSPECT_MIN_SECONDS)
        self._gate_no_face = FactTemporalGate(NO_FACE_MIN_SECONDS)
        self._gate_hand_near = FactTemporalGate(HAND_NEAR_FACE_MIN_SECONDS)

    def reset_temporal_state(self) -> None:
        """Clear temporal gates when a new monitoring session starts."""
        for gate in (
            self._gate_eyes,
            self._gate_looking_away,
            self._gate_head_down,
            self._gate_phone,
            self._gate_no_face,
            self._gate_hand_near,
        ):
            gate.reset()
        self._blink_times.clear()
        self._last_ear = 1.0
        self._ear_smoothed = 0.3
        self._ear_baseline = 0.0
        self._ear_samples = 0
        self._has_smooth = False

    def decode_frame(self, image_b64: str) -> np.ndarray | None:
        if not image_b64:
            return None
        payload = image_b64.split(",", 1)[-1]
        try:
            raw = base64.b64decode(payload)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def _smooth_pose(self, yaw: float, pitch: float) -> tuple[float, float]:
        if not self._has_smooth:
            self._smooth_yaw = yaw
            self._smooth_pitch = pitch
            self._has_smooth = True
        else:
            self._smooth_yaw += POSE_SMOOTH_ALPHA * (yaw - self._smooth_yaw)
            self._smooth_pitch += POSE_SMOOTH_ALPHA * (pitch - self._smooth_pitch)
        return self._smooth_yaw, self._smooth_pitch

    def _smooth_ear(self, ear: float) -> float:
        if self._ear_samples == 0:
            self._ear_smoothed = ear
        else:
            self._ear_smoothed += EAR_SMOOTH_ALPHA * (ear - self._ear_smoothed)
        if ear > self._ear_baseline:
            self._ear_baseline = ear
        self._ear_samples = min(self._ear_samples + 1, 120)
        return self._ear_smoothed

    def _eyes_closed_raw(self, left_ear: float, right_ear: float, smoothed: float) -> bool:
        baseline = self._ear_baseline if self._ear_baseline > 0.12 else 0.30
        adaptive_limit = max(EAR_CLOSED_THRESHOLD, baseline * EAR_CLOSED_RATIO)
        per_eye_closed = left_ear < adaptive_limit and right_ear < adaptive_limit
        avg_closed = smoothed < adaptive_limit
        both_low = min(left_ear, right_ear) < adaptive_limit
        return per_eye_closed or (avg_closed and both_low)

    def observe(self, frame: np.ndarray) -> ObservationResult:
        now = time.time()
        if frame is None or frame.size == 0:
            return ObservationResult(
                facts={"no_face": True},
                message="Invalid frame",
            )

        if not self._backend.available:
            return self._opencv_fallback(frame)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = self._backend.process_face(rgb)
        hand_result = self._backend.process_hands(rgb)

        facts: dict[str, bool] = {}
        metrics: dict[str, float] = {}
        labels: list[dict[str, Any]] = []

        if not face_result or not face_result.face_landmarks:
            self._last_draw_landmarks = None
            no_face_confirmed = self._gate_no_face.update(True, now)
            self._gate_eyes.update(False, now)
            self._gate_looking_away.update(False, now)
            self._gate_head_down.update(False, now)
            self._gate_phone.update(False, now)
            self._gate_hand_near.update(False, now)

            if no_face_confirmed:
                facts["no_face"] = True
            return ObservationResult(
                facts=facts,
                metrics={"no_face_progress": self._gate_no_face.progress(now)},
                labels=[{
                    "id": "no_face",
                    "text": "No face detected",
                    "x": 0.5,
                    "y": 0.5,
                    "tone": "danger",
                }],
                message="No face detected",
            )

        self._gate_no_face.update(False, now)
        raw_lm = face_result.face_landmarks[0]
        self._last_draw_landmarks = raw_lm
        lm = self._backend.landmarks_to_adapter(raw_lm)

        left_ear = self._ear_single(lm, [33, 160, 158, 133, 153, 144], w, h)
        right_ear = self._ear_single(lm, [362, 385, 387, 263, 373, 380], w, h)
        ear = (left_ear + right_ear) / 2.0
        smoothed_ear = self._smooth_ear(ear)

        raw_yaw, raw_pitch = self._estimate_head_pose(lm, w, h)
        yaw, pitch = self._smooth_pose(raw_yaw, raw_pitch)
        metrics["ear"] = round(smoothed_ear, 3)
        metrics["ear_left"] = round(left_ear, 3)
        metrics["ear_right"] = round(right_ear, 3)
        metrics["head_yaw"] = round(yaw, 2)
        metrics["head_pitch"] = round(pitch, 2)

        eyes_closed_raw = self._eyes_closed_raw(left_ear, right_ear, smoothed_ear)
        eyes_closed = self._gate_eyes.update(eyes_closed_raw, now)
        metrics["eyes_closed_raw"] = 1.0 if eyes_closed_raw else 0.0
        metrics["eyes_closed_progress"] = round(self._gate_eyes.progress(now), 2)

        looking_away_raw = abs(yaw) > HEAD_YAW_THRESHOLD
        head_down_raw = pitch > HEAD_PITCH_DOWN_THRESHOLD
        looking_down_raw = pitch > (HEAD_PITCH_DOWN_THRESHOLD * 0.6)

        looking_away = self._gate_looking_away.update(looking_away_raw, now)
        head_down = self._gate_head_down.update(head_down_raw, now)
        looking_down = head_down or (
            looking_down_raw and self._gate_head_down.progress(now) > 0.5
        )

        upright = (
            not head_down
            and not looking_away
            and pitch < (HEAD_PITCH_DOWN_THRESHOLD * 0.42)
            and abs(yaw) < (HEAD_YAW_THRESHOLD * 0.8)
        )
        eye_contact = (
            not eyes_closed
            and not eyes_closed_raw
            and not looking_away
            and abs(yaw) < (HEAD_YAW_THRESHOLD * 0.42)
            and not head_down
        )

        facts["face_detected"] = True
        if eyes_closed:
            facts["eyes_closed"] = True
        if looking_away:
            facts["looking_away"] = True
        if head_down:
            facts["head_down"] = True
        if looking_down:
            facts["looking_down"] = True
        if upright:
            facts["upright_posture"] = True
        if eye_contact:
            facts["eye_contact"] = True

        self._track_blink(smoothed_ear)
        if self._low_blink_rate() and not eyes_closed_raw:
            facts["low_blink_rate"] = True

        hand_near_raw, phone_sus_raw = self._analyze_hands(
            hand_landmarks_list(hand_result), lm, w, h
        )
        hand_near = self._gate_hand_near.update(hand_near_raw, now)
        phone_sus = self._gate_phone.update(phone_sus_raw, now)

        if hand_near:
            facts["hand_near_face"] = True
        if phone_sus:
            facts["phone_suspected"] = True

        labels = self._build_overlay_labels(facts, eyes_closed, eyes_closed_raw)

        return ObservationResult(
            facts=facts,
            metrics=metrics,
            labels=labels,
            message="Observations captured",
        )

    def _build_overlay_labels(
        self,
        facts: dict[str, bool],
        eyes_closed: bool,
        eyes_closed_raw: bool,
    ) -> list[dict[str, Any]]:
        """Fixed-position HUD chips on the video (not drawn on the face)."""
        labels: list[dict[str, Any]] = []
        x = 0.04
        priority = [
            ("eyes_closed", "eyes_closed", "danger"),
            ("eyes_closing", "eyes_closing", "warn"),
            ("looking_away", "looking_away", "danger"),
            ("head_down", "head_down", "warn"),
            ("phone_suspected", "phone_suspected", "danger"),
            ("hand_near_face", "hand_near_face", "warn"),
            ("eye_contact", "eye_contact", "ok"),
            ("upright_posture", "upright_posture", "ok"),
        ]
        if eyes_closed:
            labels.append({"id": "eyes_closed", "text": "eyes_closed", "x": x, "y": 0.05, "tone": "danger"})
            x += 0.14
        elif eyes_closed_raw:
            labels.append({"id": "eyes_closing", "text": "eyes_closing", "x": x, "y": 0.05, "tone": "warn"})
            x += 0.14
        for key, text, tone in priority:
            if key in ("eyes_closed", "eyes_closing"):
                continue
            if facts.get(key):
                labels.append({"id": key, "text": text, "x": min(x, 0.82), "y": 0.05, "tone": tone})
                x += 0.13
        return labels

    def annotate_frame(self, frame: np.ndarray, result: ObservationResult) -> np.ndarray:
        out = frame.copy()
        if self._last_draw_landmarks is not None:
            out = self._backend.draw_face_mesh(out, self._last_draw_landmarks)
        cv2.putText(
            out, "EduMind", (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 2, cv2.LINE_AA,
        )
        return out

    @staticmethod
    def _ear_single(lm, idx, w, h) -> float:
        p = [np.array([lm[i].x * w, lm[i].y * h]) for i in idx]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        hdist = np.linalg.norm(p[0] - p[3])
        if hdist < 1e-6:
            return 0.3
        return (v1 + v2) / (2.0 * hdist)

    def _estimate_head_pose(self, lm, w, h) -> tuple[float, float]:
        nose = np.array([lm[1].x * w, lm[1].y * h, lm[1].z * w])
        chin = np.array([lm[152].x * w, lm[152].y * h, lm[152].z * w])
        left = np.array([lm[234].x * w, lm[234].y * h])
        right = np.array([lm[454].x * w, lm[454].y * h])
        forehead = np.array([lm[10].x * w, lm[10].y * h])

        face_width = np.linalg.norm(right - left) + 1e-6
        nose_offset = (nose[0] - (left[0] + right[0]) / 2.0) / face_width
        yaw = float(np.clip(nose_offset * 110, -60, 60))

        vertical = chin[1] - forehead[1]
        pitch_ratio = (nose[1] - forehead[1]) / (vertical + 1e-6)
        pitch = float(np.clip((pitch_ratio - 0.42) * 120, -30, 60))
        return yaw, pitch

    def _track_blink(self, ear: float) -> None:
        now = time.time()
        limit = max(EAR_CLOSED_THRESHOLD, self._ear_baseline * EAR_CLOSED_RATIO)
        if self._last_ear >= limit and ear < limit:
            self._blink_times.append(now)
        self._last_ear = ear
        self._blink_times = [t for t in self._blink_times if now - t < 60.0]

    def _low_blink_rate(self) -> bool:
        if self._ear_samples < 30:
            return False
        limit = max(EAR_CLOSED_THRESHOLD, self._ear_baseline * EAR_CLOSED_RATIO)
        return len(self._blink_times) < 2 and self._last_ear > limit

    def _analyze_hands(self, hands, lm, w, h) -> tuple[bool, bool]:
        if not hands:
            return False, False

        face_y = np.mean([lm[i].y for i in (10, 152, 1)]) * h
        face_x = lm[1].x * w
        face_w = abs(lm[454].x - lm[234].x) * w
        hand_near = False
        phone_sus = False
        near_limit = max(h * HAND_NEAR_FACE_RATIO, face_w * 0.9)

        for hand_lm in hands:
            index_tip = hand_lm[8]
            wrist = hand_lm[0]
            hx = (index_tip.x + wrist.x) / 2 * w
            hy = (index_tip.y + wrist.y) / 2 * h
            dist = np.hypot(hx - face_x, hy - face_y)
            if dist < near_limit:
                hand_near = True
            if hy > face_y + h * 0.05 and dist < h * PHONE_BELOW_FACE_RATIO:
                phone_sus = True

        return hand_near, phone_sus

    def _opencv_fallback(self, frame: np.ndarray) -> ObservationResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        facts: dict[str, bool] = {}
        if len(faces) == 0:
            facts["no_face"] = True
            return ObservationResult(facts=facts, message="Fallback: no face")
        facts["face_detected"] = True
        facts["eye_contact"] = True
        facts["upright_posture"] = True
        return ObservationResult(
            facts=facts,
            metrics={"mode": 0},
            labels=[{"id": "posture", "text": "Posture: upright (limited)", "x": 0.5, "y": 0.12, "tone": "ok"}],
            message="OpenCV Haar fallback (limited cues)",
        )
