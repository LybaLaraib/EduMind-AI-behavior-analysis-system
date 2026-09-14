"""
Prolog expert engine — knowledge base loading, fact assertion, backward-chaining inference.
Falls back to an embedded Horn-clause evaluator when SWI-Prolog is unavailable (dev only).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any

from config import PROLOG_KB_PATH

OBSERVATION_ATOMS = (
    "face_detected",
    "no_face",
    "eyes_closed",
    "eye_contact",
    "looking_away",
    "looking_down",
    "head_down",
    "upright_posture",
    "phone_suspected",
    "hand_near_face",
    "low_blink_rate",
)


class EmbeddedRuleEngine:
    """Mirrors knowledge_base.pl for environments without SWI-Prolog."""

    def clear_observations(self) -> None:
        self._obs: set[str] = set()

    def assert_observation(self, fact: str) -> None:
        if fact in OBSERVATION_ATOMS:
            self._obs.add(fact)

    def infer_all(self) -> dict[str, Any]:
        obs = self._obs
        sleepy = ("eyes_closed" in obs and "head_down" in obs) or (
            "eyes_closed" in obs and "low_blink_rate" in obs
        )
        drowsy_eyes = (
            "eyes_closed" in obs
            and "head_down" not in obs
            and "phone_suspected" not in obs
        )
        distracted = (
            ("looking_away" in obs and "face_detected" in obs)
            or ("looking_away" in obs and "phone_suspected" in obs)
            or (
                "looking_away" in obs
                and "head_down" in obs
                and "eyes_closed" not in obs
            )
        )
        phone = "phone_suspected" in obs or (
            "hand_near_face" in obs and "looking_down" in obs
        )
        attentive = (
            "face_detected" in obs
            and "eye_contact" in obs
            and "upright_posture" in obs
            and "looking_away" not in obs
            and "eyes_closed" not in obs
            and "phone_suspected" not in obs
        )
        disengaged = "no_face" in obs and "face_detected" not in obs
        partial = (
            "face_detected" in obs
            and not attentive
            and not sleepy
            and not drowsy_eyes
            and not distracted
            and not phone
        )

        applied: list[dict[str, str]] = [{
            "technique": "Unification",
            "rule": "assert_observation/1",
            "result": f"Facts bound: {', '.join(sorted(obs)) or 'none'}.",
        }]

        if attentive:
            applied.extend([
                {"technique": "Backward chaining", "rule": "attentive_student",
                 "result": "Behavior classified as attentive."},
                {"technique": "Modus ponens", "rule": "attention_level(high) :- attentive_student",
                 "result": "Attention level set to high."},
            ])
            attention, behavior, risk = "high", "attentive", "low"
            rec = "Maintain your current focus — excellent engagement with the lecture."
            reason = "Stable posture, open eyes, and eye contact with the display."
        elif sleepy and not phone:
            applied.append({"technique": "Backward chaining", "rule": "sleepy_student",
                            "result": "Drowsiness inferred from eyes and posture."})
            attention, behavior, risk = "low", "sleepy", "medium"
            rec = "You appear tired. Consider a short break or adjust your seating for alertness."
            reason = "Eyes closed with head lowered — indicators of drowsiness."
        elif drowsy_eyes and not phone:
            applied.append({"technique": "Backward chaining", "rule": "drowsy_eyes",
                            "result": "Eyes closed 2+ seconds — attention lowered."})
            attention, behavior, risk = "low", "sleepy", "medium"
            rec = "Your eyes have been closed — reopen them and refocus on the screen."
            reason = "Eyes closed for an extended period — attention is reduced."
        elif phone:
            applied.append({"technique": "Backward chaining", "rule": "phone_usage_suspected",
                            "result": "Possible mobile device use detected."})
            attention, behavior, risk = "low", "phone_use", "medium"
            rec = "Put your phone away and return attention to the learning material."
            reason = "Hand activity near face suggests possible mobile device use."
        elif distracted and not sleepy:
            applied.append({"technique": "Backward chaining", "rule": "distracted_student",
                            "result": "Distraction inferred from gaze/posture."})
            attention, behavior, risk = "medium", "distracted", "medium"
            rec = "Refocus on the screen. Minimize side glances and keep notes within your primary view."
            reason = "Frequent gaze away from the learning screen."
        elif disengaged:
            applied.append({"technique": "Backward chaining", "rule": "disengaged_student",
                            "result": "No face visible."})
            attention, behavior, risk = "low", "disengaged", "high"
            rec = "Position yourself in front of the camera so the system can support your session."
            reason = "No face visible — student may have left the seat or camera is blocked."
        elif partial:
            applied.append({"technique": "Unification", "rule": "partially_attentive",
                            "result": "Mixed signals — monitoring continues."})
            attention, behavior, risk = "medium", "neutral", "low"
            rec = "Stay engaged — re-center your gaze on the instructor or slides."
            reason = "Mixed signals — monitor for sustained patterns."
        else:
            attention, behavior, risk = "medium", "neutral", "low"
            rec = "Stay engaged — re-center your gaze on the instructor or slides."
            reason = "Mixed signals — monitor for sustained patterns."

        if sleepy and phone:
            risk = "high"
            attention = "low"

        return {
            "attention": attention,
            "behavior": behavior,
            "risk": risk,
            "recommendation": rec,
            "reason": reason,
            "flags": {"sleepy": sleepy, "distracted": distracted, "phone": phone, "attentive": attentive},
            "inference_steps": applied,
            "engine": "embedded",
        }


class PrologExpertEngine:
    def __init__(self, kb_path: str | None = None):
        self.kb_path = kb_path or PROLOG_KB_PATH
        self._prolog = None
        self._embedded: EmbeddedRuleEngine | None = None
        self._mode = "unknown"
        self._init_engine()

    def _init_engine(self) -> None:
        try:
            from pyswip import Prolog

            self._prolog = Prolog()
            self._prolog.consult(os.path.abspath(self.kb_path))
            list(self._prolog.query("load_expert_rules"))
            self._mode = "pyswip"
        except Exception:
            if self._try_swipl_subprocess():
                self._mode = "swipl_cli"
            else:
                self._embedded = EmbeddedRuleEngine()
                self._mode = "embedded"

    def _try_swipl_subprocess(self) -> bool:
        try:
            subprocess.run(
                ["swipl", "--version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except Exception:
            return False

    @property
    def engine_mode(self) -> str:
        return self._mode

    def clear_observations(self) -> None:
        if self._embedded:
            self._embedded.clear_observations()
            return
        if self._prolog:
            list(self._prolog.query("clear_observations"))

    def assert_observations(self, facts: dict[str, bool]) -> list[str]:
        active = []
        self.clear_observations()
        for atom in OBSERVATION_ATOMS:
            if facts.get(atom):
                active.append(atom)
                if self._embedded:
                    self._embedded.assert_observation(atom)
                elif self._prolog:
                    list(self._prolog.query(f"assert_observation({atom})"))
        self._pending_obs = active
        return active

    def infer_all(self) -> dict[str, Any]:
        if self._embedded:
            result = self._embedded.infer_all()
            result["engine"] = "embedded"
            result["observations"] = sorted(self._embedded._obs)
            return result

        if self._mode == "swipl_cli":
            return self._infer_via_cli()

        if not self._prolog:
            raise RuntimeError("Prolog engine not initialized")

        rows = list(self._prolog.query("infer_all(Results)"))
        if not rows:
            return self._default_result()

        raw = rows[0].get("Results") or rows[0].get("results")
        return self._normalize_prolog_result(raw)

    def _infer_via_cli(self) -> dict[str, Any]:
        script = self._build_cli_script()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pl", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = subprocess.run(
                ["swipl", "-q", "-s", script_path, "-g", "main,halt(0)"],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=os.path.dirname(self.kb_path),
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr or proc.stdout)
            data = json.loads(proc.stdout.strip() or "{}")
            data["engine"] = "swipl_cli"
            return data
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def _build_cli_script(self) -> str:
        obs_lines = getattr(self, "_pending_obs", [])
        asserts = "\n".join(f"assert_observation({o})." for o in obs_lines) or "true"
        kb = self.kb_path.replace("\\", "/")
        return f"""
:- use_module(library(http/json)).
:- consult('{kb}').
main :-
    clear_observations,
    {asserts},
    infer_all(R),
    json_write_dict(current_output, R, []).
"""

    def set_pending_observations(self, obs: list[str]) -> None:
        self._pending_obs = obs

    def _normalize_prolog_result(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            flags = raw.get("flags") or {}
            if hasattr(flags, "__dict__"):
                flags = dict(flags)
            steps = raw.get("inference_steps") or []
            return {
                "attention": self._unwrap(raw.get("attention"), "medium"),
                "behavior": self._unwrap(raw.get("behavior"), "neutral"),
                "risk": self._unwrap(raw.get("risk"), "low"),
                "recommendation": self._unwrap(
                    raw.get("recommendation"),
                    "Continue monitoring your head posture.",
                ),
                "reason": self._unwrap(raw.get("reason"), "Analysis complete."),
                "flags": flags if isinstance(flags, dict) else {},
                "inference_steps": steps if isinstance(steps, list) else [],
                "engine": self._mode,
            }
        return self._default_result()

    @staticmethod
    def _unwrap(val: Any, default: str) -> str:
        if val is None:
            return default
        if isinstance(val, (str, int, float, bool)):
            return str(val)
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        return str(val)

    @staticmethod
    def _default_result() -> dict[str, Any]:
        return {
            "attention": "medium",
            "behavior": "neutral",
            "risk": "low",
            "recommendation": "Continue monitoring your head posture.",
            "reason": "Insufficient data for a firm conclusion.",
            "flags": {},
            "inference_steps": [],
            "engine": "unknown",
        }

    def get_rule_catalog(self) -> list[dict[str, str]]:
        return [
            {
                "id": "R1",
                "name": "sleepy_student",
                "type": "backward chaining",
                "rule": "sleepy_student :- eyes_closed, head_down.",
                "description": "Infers drowsiness when eyes are closed and head is lowered.",
            },
            {
                "id": "R2",
                "name": "distracted_student",
                "type": "backward chaining",
                "rule": "distracted_student :- looking_away, phone_suspected.",
                "description": "Infers distraction from gaze away plus suspected phone use.",
            },
            {
                "id": "R3",
                "name": "attentive_student",
                "type": "backward chaining",
                "rule": "attentive_student :- eye_contact, upright_posture, \\+ eyes_closed, \\+ phone_suspected.",
                "description": "Infers high engagement when posture, gaze, and eyes align.",
            },
            {
                "id": "R4",
                "name": "attention_level(high)",
                "type": "modus ponens",
                "rule": "attention_level(high) :- attentive_student.",
                "description": "Maps attentive behavior to high attention level.",
            },
            {
                "id": "R5",
                "name": "phone_usage_suspected",
                "type": "backward chaining",
                "rule": "phone_usage_suspected :- hand_near_face, looking_down.",
                "description": "Heuristic for mobile device use from hand and gaze cues.",
            },
            {
                "id": "R6",
                "name": "disengaged_student",
                "type": "expert system",
                "rule": "disengaged_student :- no_face, \\+ face_detected.",
                "description": "Flags disengagement when no face is visible.",
            },
        ]
