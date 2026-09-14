import logging

from flask import Blueprint, jsonify, render_template, request

from app.services.analyzer import SessionAnalyzer

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)
analyzer = SessionAnalyzer()


@bp.route("/")
def index():
    return render_template("index.html", info=analyzer.get_system_info())


@bp.route("/monitor")
def monitor():
    return render_template(
        "monitor.html",
        info=analyzer.get_system_info(),
        page_class="monitor-page",
    )


@bp.route("/knowledge")
def knowledge():
    return render_template(
        "knowledge.html",
        rules=analyzer.get_rules(),
        info=analyzer.get_system_info(),
    )


@bp.route("/about")
def about():
    return render_template("about.html", info=analyzer.get_system_info())


@bp.route("/report")
def report():
    summary = analyzer.get_session_summary()
    return render_template(
        "report.html",
        summary=summary,
        info=analyzer.get_system_info(),
    )


@bp.route("/api/session/start", methods=["POST"])
def api_session_start():
    return jsonify(analyzer.start_session())


@bp.route("/api/session/stop", methods=["POST"])
def api_session_stop():
    return jsonify(analyzer.stop_session())


@bp.route("/api/session/summary")
def api_session_summary():
    return jsonify(analyzer.get_session_summary())


@bp.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    image = data.get("image", "")
    annotate = data.get("annotate", True)
    if not image:
        return jsonify({"ok": False, "error": "No image provided"}), 400
    try:
        result = analyzer.analyze_frame(image, annotate=annotate)
    except Exception as exc:
        logger.exception("analyze_frame failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    status = 200 if result.get("ok") else 400
    return jsonify(result), status
