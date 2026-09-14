import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

SECRET_KEY = os.environ.get("EDUMIND_SECRET_KEY", "edumind-dev-secret-change-in-production")

PROLOG_KB_PATH = os.path.join(BASE_DIR, "prolog", "knowledge_base.pl")

# Observation thresholds (tuned for typical laptop webcam)
EAR_CLOSED_THRESHOLD = 0.24
EAR_CLOSED_RATIO = 0.72
EAR_SMOOTH_ALPHA = 0.4
HEAD_YAW_THRESHOLD = 22.0
HEAD_PITCH_DOWN_THRESHOLD = 16.0
HAND_NEAR_FACE_RATIO = 0.38
PHONE_BELOW_FACE_RATIO = 0.42

# Frame capture / analysis interval (seconds) — synced with frontend capture rate
FRAME_INTERVAL_SEC = 1.0

# Temporal fact confirmation (seconds of continuous raw detection)
EYES_CLOSED_MIN_SECONDS = 2.0
LOOKING_AWAY_MIN_SECONDS = 1.2
HEAD_DOWN_MIN_SECONDS = 1.2
PHONE_SUSPECT_MIN_SECONDS = 1.5
NO_FACE_MIN_SECONDS = 1.0
HAND_NEAR_FACE_MIN_SECONDS = 1.0

# Session aggregation
SESSION_HISTORY_LIMIT = 180
