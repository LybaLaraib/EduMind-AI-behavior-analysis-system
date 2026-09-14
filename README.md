# EduMind — Intelligent Classroom Behavior Expert System

**EduMind** is an AI lab final project that combines a **classical expert system** (Prolog) with **computer vision observations** (OpenCV + MediaPipe). The camera does not make decisions — it only supplies facts. Prolog applies rules and performs backward-chaining inference to conclude attention level, behavior, risk, and recommendations.

📄 [Full Project Report](AI%20Terminal%20Proj%20Report.pdf)

## Problem Statement

Online and hybrid students often disengage silently (gaze away, fatigue, phone use). EduMind acts as a smart classroom observer: it watches behavior, represents knowledge as logical facts, applies expert rules, and explains its conclusions — matching core AI course topics (knowledge representation, rules, inference, reasoning).

## AI Concepts Covered

| Concept | Where |
|--------|--------|
| Knowledge representation | Dynamic `observation/1` facts |
| Rules | Horn clauses in `prolog/knowledge_base.pl` |
| Inference | Prolog backward chaining (SLD) |
| Expert system | CV → facts → KB → conclusions → advice |
| Reasoning | Modus ponens style rules (e.g. attentive → high attention) |
| Computer vision | OpenCV + MediaPipe (observations only) |

## Architecture

```
Browser GUI → Flask API → OpenCV/MediaPipe (facts) → Prolog (inference) → Results
```

## Project Structure

```
Lab Final Project/
├── run.py                 # Start server
├── config.py
├── requirements.txt
├── prolog/
│   └── knowledge_base.pl  # Expert rules
├── app/
│   ├── routes.py
│   ├── cv/behavior_observer.py
│   ├── expert/prolog_engine.py
│   └── services/analyzer.py
├── templates/             # HTML pages
└── static/                # CSS, JS
```

## Requirements

- Python 3.10+
- Webcam (for live monitor)
- **SWI-Prolog** (recommended for viva — real Prolog inference)
  - Download: https://www.swi-prolog.org/download/stable
  - Add `swipl` to system PATH

## Installation

```bash
cd "Lab Final Project"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python run.py
```

Open **http://127.0.0.1:5000** in Chrome/Edge, allow camera access on **Live Monitor**.

## Pages

- **Home** — overview and architecture
- **Live Monitor** — webcam + real-time expert analysis
- **Session Report** — attention timeline chart
- **Knowledge Base** — Prolog rules for viva
- **About AI** — problem scope and concept mapping

## Prolog Engine Modes

1. **pyswip** — if SWI-Prolog + pyswip work
2. **swipl_cli** — subprocess fallback
3. **embedded** — Python mirror of rules (dev/demo if Prolog missing)

For demonstrations and viva, install SWI-Prolog so mode shows `pyswip` or `swipl_cli`.

## Viva Talking Points

1. **OpenCV does not decide** — only extracts facts (`eyes_closed`, `looking_away`, …).
2. **Prolog decides** — rules like `sleepy_student :- eyes_closed, head_down.`
3. **Backward chaining** — goals such as `attention_level(X)` proven from rules.
4. **Separation of concerns** — expert system architecture (knowledge, inference, explanation).

## Authors

AI Lab Final Project — Artificial Intelligence Course
