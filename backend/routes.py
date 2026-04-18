from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request

from .services.frontend_service import render_frontend
from .services.report_service import (
    build_analysis_report,
    get_report,
    list_reports,
    remove_upload,
    save_report,
    save_upload,
    validate_patient_payload,
)


api_blueprint = Blueprint("api", __name__)


def _model_service():
    return current_app.extensions["model_service"]


@api_blueprint.get("/")
def serve_frontend():
    html = render_frontend(
        frontend_path=current_app.config["BASE_DIR"] / current_app.config["FRONTEND_FILE"],
    )
    return Response(html, mimetype="text/html")


@api_blueprint.get("/api/health")
def health():
    status = _model_service().health_status()
    status["frontend_file"] = current_app.config["FRONTEND_FILE"]
    return jsonify(status)


@api_blueprint.get("/api/analyses")
def analyses_index():
    limit = request.args.get("limit", default=20, type=int)
    reports = list_reports(current_app.config["REPORT_DIR"], limit=limit)
    return jsonify({"items": reports, "count": len(reports)})


@api_blueprint.get("/api/analyses/<analysis_id>")
def analyses_detail(analysis_id: str):
    report = get_report(current_app.config["REPORT_DIR"], analysis_id)
    if report is None:
        return jsonify({"error": "Analysis report not found."}), 404
    return jsonify(report)


@api_blueprint.post("/api/analyze")
@api_blueprint.post("/api/predict")
@api_blueprint.post("/predict")
def analyze():
    try:
        patient = validate_patient_payload(request.form)
        image_file = request.files.get("image") or request.files.get("file")
        if image_file is None:
            return jsonify({"error": "No image uploaded."}), 400

        upload_meta = save_upload(
            image_file=image_file,
            upload_dir=current_app.config["UPLOAD_DIR"],
            allowed_extensions=current_app.config["ALLOWED_EXTENSIONS"],
        )

        try:
            prediction = _model_service().predict(upload_meta["path"])
            report = build_analysis_report(
                patient=patient,
                upload_meta=upload_meta,
                prediction=prediction,
            )
            save_report(report=report, report_dir=current_app.config["REPORT_DIR"])
        finally:
            remove_upload(Path(upload_meta["path"]))

        return jsonify(report)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        current_app.logger.exception("Unexpected analysis failure")
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500
