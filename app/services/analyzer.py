"""
Orchestrates CV observations → Prolog inference → session analytics.
"""

from __future__ import annotations

import base64
import threading
import time
from collections import Counter, deque
from typing import Any

import cv2

from app.cv.behavior_observer import BehaviorObserver
from app.cv.fact_temporal import FactSessionAccumulator
from app.expert.prolog_engine import PrologExpertEngine
from config import FRAME_INTERVAL_SEC, SESSION_HISTORY_LIMIT


class SessionAnalyzer:
    def __init__(self):
        self.observer = BehaviorObserver()
        self.expert = PrologExpertEngine()
        self.history: deque[dict[str, Any]] = deque(maxlen=SESSION_HISTORY_LIMIT)
        self.fact_accumulator = FactSessionAccumulator()
        self.session_active = False
        self.session_start: float | None = None
        self.frame_count = 0
        self._last_frame_time: float | None = None
        self._analyze_lock = threading.Lock()

    def start_session(self) -> dict[str, Any]:
        self.session_active = True
        self.session_start = time.time()
        self._last_frame_time = self.session_start
        self.history.clear()
        self.frame_count = 0
        self.observer.reset_temporal_state()
        self.fact_accumulator.reset()
        return {
            "active": True,
            "engine": self.expert.engine_mode,
            "started_at": self.session_start,
            "frame_interval_sec": FRAME_INTERVAL_SEC,
        }

    def stop_session(self) -> dict[str, Any]:
        self.session_active = False
        summary = self.get_session_summary()
        summary["active"] = False
        return summary

    def analyze_frame(self, image_b64: str, annotate: bool = True) -> dict[str, Any]:
        with self._analyze_lock:
            return self._analyze_frame_locked(image_b64, annotate)

    def _analyze_frame_locked(self, image_b64: str, annotate: bool) -> dict[str, Any]:
        frame = self.observer.decode_frame(image_b64)
        if frame is None:
            return {"ok": False, "error": "Could not decode image frame"}

        now = time.time()
        dt = FRAME_INTERVAL_SEC
        if self._last_frame_time is not None:
            dt = max(FRAME_INTERVAL_SEC * 0.85, now - self._last_frame_time)
        self._last_frame_time = now

        observation = self.observer.observe(frame)
        active_facts = self.expert.assert_observations(observation.facts)

        if self.expert.engine_mode == "swipl_cli":
            self.expert.set_pending_observations(active_facts)

        inference = self.expert.infer_all()
        inference["observations"] = active_facts

        attention_score = self._attention_score(
            inference, observation.metrics, observation.facts
        )
        fact_snapshot = {k: True for k, v in observation.facts.items() if v}
        record = {
            "timestamp": now,
            "observations": active_facts,
            "fact_snapshot": fact_snapshot,
            "metrics": observation.metrics,
            "inference": inference,
            "attention_score": attention_score,
            "behavior": inference.get("behavior", "neutral"),
        }

        if self.session_active:
            self.fact_accumulator.tick(observation.facts, dt)
            self.history.append(record)
        self.frame_count += 1

        preview_b64 = None
        if annotate:
            annotated = self.observer.annotate_frame(frame, observation)
            _, buf = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 72]
            )
            preview_b64 = "data:image/jpeg;base64," + base64.b64encode(
                buf
            ).decode("ascii")

        session_facts = None
        if self.session_active:
            session_facts = self.fact_accumulator.get_summary()

        return {
            "ok": True,
            "observation": observation.to_dict(),
            "inference": inference,
            "attention_score": attention_score,
            "preview": preview_b64,
            "labels": observation.labels,
            "engine": self.expert.engine_mode,
            "frame": self.frame_count,
            "session_facts": session_facts,
            "frame_interval_sec": FRAME_INTERVAL_SEC,
        }

    def get_session_summary(self) -> dict[str, Any]:
        empty = {
            "frames": 0,
            "average_attention": 0,
            "dominant_behavior": "none",
            "timeline": [],
            "duration_seconds": 0,
            "duration_label": "0 sec",
            "behavior_stats": {},
            "fact_session_totals": {},
            "inference_timeline": [],
            "report_methodology": [
                "Run a live monitoring session to populate full behavior analytics.",
                "CV detects posture, eyes, hands/mobile, and face presence each frame.",
                "Prolog rules infer sleepy, distracted, attentive, phone_use, and disengaged states.",
            ],
        }
        if not self.history:
            return empty

        scores = [h["attention_score"] for h in self.history]
        behaviors = [h.get("behavior", "neutral") for h in self.history]
        dominant = Counter(behaviors).most_common(1)[0][0]

        duration = len(self.history) * FRAME_INTERVAL_SEC
        if len(self.history) >= 2:
            ts_span = self.history[-1]["timestamp"] - self.history[0]["timestamp"]
            duration = max(duration, ts_span + FRAME_INTERVAL_SEC)
        elif len(self.history) == 1:
            duration = max(duration, FRAME_INTERVAL_SEC)

        behavior_stats = self._behavior_stats(duration)
        inference_timeline = self._inference_timeline()
        fact_session_totals = self._format_fact_session_totals()

        return {
            "frames": len(self.history),
            "average_attention": round(sum(scores) / len(scores), 1),
            "dominant_behavior": dominant,
            "duration_seconds": round(duration, 1),
            "duration_label": self._format_duration(duration),
            "behavior_stats": behavior_stats,
            "fact_session_totals": fact_session_totals,
            "inference_timeline": inference_timeline,
            "report_methodology": [
                "OpenCV + MediaPipe extract posture, eye openness (EAR), hands, and face presence each frame.",
                "Temporal gating: eyes_closed requires 2+ seconds continuous closure; other facts use hold timers.",
                "Session totals sum every confirmed period (seconds and frames) across the full session.",
                "Facts asserted: looking_away, head_down, eyes_closed, phone_suspected, no_face, etc.",
                "Backward chaining proves sleepy_student, distracted_student, attentive_student, phone_usage.",
                "Modus ponens maps proven behaviors to attention_level conclusions.",
                "Unification binds variables during SLD resolution across Horn clauses.",
                "The expert system produces behavior, risk, recommendation, and explanation.",
                "Python aggregates behaviors and observation counts into session statistics.",
            ],
            "timeline": [
                {
                    "t": round(i * FRAME_INTERVAL_SEC, 1),
                    "score": h["attention_score"],
                    "behavior": h.get("behavior", "neutral"),
                }
                for i, h in enumerate(self.history)
            ],
        }

    @staticmethod
    def _record_has_fact(record: dict[str, Any], fact: str) -> bool:
        if fact in record.get("observations", []):
            return True
        return bool(record.get("fact_snapshot", {}).get(fact))

    @staticmethod
    def _record_behavior(record: dict[str, Any], *behaviors: str) -> bool:
        b = record.get("behavior") or record.get("inference", {}).get("behavior")
        return b in behaviors

    def _count_matching_frames(self, predicate) -> int:
        return sum(1 for h in self.history if predicate(h))

    def _duration_for_predicate(self, predicate) -> float:
        if not self.history:
            return 0.0
        total = 0.0
        history_list = list(self.history)
        for i, record in enumerate(history_list):
            if not predicate(record):
                continue
            if i + 1 < len(history_list):
                dt = history_list[i + 1]["timestamp"] - record["timestamp"]
            else:
                dt = FRAME_INTERVAL_SEC
            total += max(dt, FRAME_INTERVAL_SEC * 0.85)
        return total

    def _stat_from_accumulator(
        self, key: str, label: str, total_frames: int
    ) -> dict[str, Any]:
        raw = self.fact_accumulator.stats.get(key, {})
        frames = int(raw.get("frames", 0))
        seconds = float(raw.get("seconds", 0.0))
        periods = int(raw.get("periods", 0))
        return {
            "label": label,
            "frames": frames,
            "seconds": round(seconds, 1),
            "duration_label": self._format_duration(seconds, frames),
            "percent": round(100 * frames / total_frames, 1) if total_frames else 0,
            "periods": periods,
        }

    def _stat_from_history_predicate(
        self, label: str, predicate, total_frames: int, duration: float
    ) -> dict[str, Any]:
        frames = self._count_matching_frames(predicate)
        seconds = round(self._duration_for_predicate(predicate), 1)
        if frames > 0 and seconds < 0.1:
            seconds = round(frames * FRAME_INTERVAL_SEC, 1)
        return {
            "label": label,
            "frames": frames,
            "seconds": seconds,
            "duration_label": self._format_duration(seconds, frames),
            "percent": round(100 * frames / total_frames, 1) if total_frames else 0,
        }

    def _merged_frame_count(self, keys: tuple[str, ...], behaviors: tuple[str, ...] = ()) -> int:
        count = 0
        for h in self.history:
            snap = h.get("fact_snapshot", {})
            if any(snap.get(k) for k in keys):
                count += 1
                continue
            b = h.get("behavior") or h.get("inference", {}).get("behavior")
            if b in behaviors:
                count += 1
        return count

    def _behavior_stats(self, duration: float) -> dict[str, Any]:
        total_frames = len(self.history)
        acc = self.fact_accumulator.stats

        def focused(r: dict) -> bool:
            return self._record_behavior(r, "attentive") or (
                self._record_has_fact(r, "eye_contact")
                and self._record_has_fact(r, "upright_posture")
                and not self._record_has_fact(r, "eyes_closed")
            )

        def not_attentive(r: dict) -> bool:
            return (
                self._record_behavior(r, "disengaged", "neutral", "distracted", "sleepy", "phone_use")
                or self._record_has_fact(r, "eyes_closed")
                or h_inference_low(r)
            ) and not focused(r)

        def h_inference_low(r: dict) -> bool:
            return r.get("inference", {}).get("attention") == "low"

        def sleepy_hist(r: dict) -> bool:
            return (
                self._record_has_fact(r, "eyes_closed")
                or self._record_behavior(r, "sleepy")
                or r.get("inference", {}).get("behavior") == "sleepy"
            )

        def diverged_hist(r: dict) -> bool:
            return (
                self._record_has_fact(r, "looking_away")
                or self._record_has_fact(r, "head_down")
                or self._record_behavior(r, "distracted")
            )

        def phone_hist(r: dict) -> bool:
            return (
                self._record_has_fact(r, "phone_suspected")
                or self._record_behavior(r, "phone_use")
            )

        eyes_stat = self._stat_from_accumulator(
            "eyes_closed", "Eyes closed (2s+ confirmed)", total_frames
        )
        away_stat = self._stat_from_accumulator("looking_away", "Looking away", total_frames)
        down_stat = self._stat_from_accumulator("head_down", "Head lowered", total_frames)
        phone_acc = self._stat_from_accumulator(
            "phone_suspected", "Phone use suspected", total_frames
        )

        diverged_frames = self._merged_frame_count(
            ("looking_away", "head_down"), ("distracted",)
        )
        diverged_seconds = max(
            float(away_stat["seconds"]) + float(down_stat["seconds"]),
            diverged_frames * FRAME_INTERVAL_SEC * 0.85,
        )
        if diverged_frames > 0:
            diverged_seconds = min(diverged_seconds, duration)

        sleepy_frames = max(
            int(eyes_stat["frames"]),
            self._count_matching_frames(sleepy_hist),
        )
        sleepy_seconds = max(float(eyes_stat["seconds"]), sleepy_frames * FRAME_INTERVAL_SEC * 0.85)

        phone_frames = max(int(phone_acc["frames"]), self._count_matching_frames(phone_hist))

        return {
            "focused": self._stat_from_history_predicate(
                "Focused (attentive)", focused, total_frames, duration
            ),
            "diverged": {
                "label": "Diverged (posture / distracted)",
                "frames": diverged_frames,
                "seconds": round(diverged_seconds, 1),
                "duration_label": self._format_duration(diverged_seconds, diverged_frames),
                "percent": round(100 * diverged_frames / total_frames, 1) if total_frames else 0,
                "periods": int(acc.get("looking_away", {}).get("periods", 0))
                + int(acc.get("head_down", {}).get("periods", 0)),
            },
            "sleepy": {
                "label": "Sleepy / drowsy",
                "frames": sleepy_frames,
                "seconds": round(sleepy_seconds, 1),
                "duration_label": self._format_duration(sleepy_seconds, sleepy_frames),
                "percent": round(100 * sleepy_frames / total_frames, 1) if total_frames else 0,
                "periods": int(eyes_stat.get("periods", 0)),
            },
            "phone_use": {
                **phone_acc,
                "frames": max(int(phone_acc["frames"]), phone_frames),
                "percent": round(
                    100 * max(int(phone_acc["frames"]), phone_frames) / total_frames, 1
                )
                if total_frames
                else 0,
            },
            "eyes_closed": eyes_stat,
            "not_attentive": self._stat_from_history_predicate(
                "Not attentive / low attention", not_attentive, total_frames, duration
            ),
        }

    def _inference_timeline(self) -> list[dict[str, Any]]:
        seen: set[str] = set()
        timeline: list[dict[str, Any]] = []
        for i, record in enumerate(self.history):
            steps = record["inference"].get("inference_steps") or []
            for step in steps:
                if not isinstance(step, dict):
                    continue
                key = f"{step.get('rule')}|{step.get('result')}"
                if key in seen:
                    continue
                seen.add(key)
                timeline.append({
                    "t": round(i * FRAME_INTERVAL_SEC, 1),
                    "technique": step.get("technique", "Inference"),
                    "rule": step.get("rule", "—"),
                    "result": step.get("result", "—"),
                })
        return timeline[:24]

    def _format_fact_session_totals(self) -> dict[str, dict[str, Any]]:
        labels = {
            "eyes_closed": "Eyes closed (held 2+ seconds)",
            "looking_away": "Looking away",
            "head_down": "Head down",
            "phone_suspected": "Phone suspected",
            "no_face": "No face detected",
        }
        totals: dict[str, dict[str, Any]] = {}
        total_frames = len(self.history) or 1
        for key, label in labels.items():
            raw = self.fact_accumulator.stats.get(key, {})
            frames = int(raw.get("frames", 0))
            seconds = float(raw.get("seconds", 0.0))
            periods = int(raw.get("periods", 0))
            totals[key] = {
                "label": label,
                "frames": frames,
                "seconds": round(seconds, 1),
                "periods": periods,
                "duration_label": self._format_duration(seconds, frames),
                "percent": round(100 * frames / total_frames, 1),
            }
        return totals

    @staticmethod
    def _format_duration(seconds: float, frames: int = 0) -> str:
        if frames > 0 and seconds < 1:
            return f"{max(seconds, FRAME_INTERVAL_SEC * 0.85):.1f} sec"
        if seconds < 60:
            if seconds < 1 and frames == 0:
                return "0 sec"
            return f"{seconds:.1f} sec".replace(".0 sec", " sec")
        minutes = int(seconds // 60)
        rem = int(round(seconds % 60))
        if rem == 0:
            return f"{minutes} min"
        return f"{minutes} min {rem} sec"

    @staticmethod
    def _attention_score(
        inference: dict, metrics: dict, facts: dict[str, bool] | None = None
    ) -> float:
        facts = facts or {}
        level = inference.get("attention", "medium")
        behavior = inference.get("behavior", "neutral")
        base = {"high": 88.0, "medium": 58.0, "low": 28.0}.get(level, 50.0)

        if facts.get("eyes_closed") or behavior == "sleepy":
            base = min(base, 22.0)
        elif metrics.get("eyes_closed_raw", 0) > 0:
            progress = float(metrics.get("eyes_closed_progress", 0))
            base -= min(20, 8 + progress * 12)

        risk = inference.get("risk", "low")
        if risk == "high":
            base -= 18
        elif risk == "medium":
            base -= 8
        yaw = abs(metrics.get("head_yaw", 0))
        pitch = metrics.get("head_pitch", 0)
        if yaw > 20:
            base -= min(15, yaw * 0.35)
        if pitch > 15:
            base -= min(12, (pitch - 10) * 0.4)
        if yaw < 8 and pitch < 10 and not facts.get("eyes_closed"):
            base += 5
        ear = metrics.get("ear")
        if ear is not None and not facts.get("eyes_closed"):
            if ear < 0.18:
                base -= 10
            elif ear > 0.26:
                base += 4
        return float(max(5.0, min(98.0, round(base, 1))))

    def get_rules(self) -> list[dict[str, str]]:
        return self.expert.get_rule_catalog()

    def get_system_info(self) -> dict[str, Any]:
        return {
            "name": "EduMind",
            "tagline": "Intelligent Classroom Behavior Expert System",
            "engine": self.expert.engine_mode,
            "cv": (
                "OpenCV + MediaPipe Tasks (face mesh, hands)"
                if self.observer._backend.available
                else "OpenCV fallback (install models — see README)"
            ),
            "ai": "Prolog (backward chaining, rule-based expert system)",
        }
