from flask import Flask

from .config import Config
from .routes import api_blueprint
from .services.model_service import ModelService


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    app.json.sort_keys = False

    for key in ("DATA_DIR", "UPLOAD_DIR", "REPORT_DIR"):
        app.config[key].mkdir(parents=True, exist_ok=True)

    model_service = ModelService(
        model_paths=app.config["MODEL_PATHS"],
        image_size=app.config["IMG_SIZE"],
        class_names=app.config["CLASSES"],
        diagnosis_info=app.config["DIAGNOSIS_INFO"],
    )

    app.extensions["model_service"] = model_service
    app.register_blueprint(api_blueprint)
    return app
