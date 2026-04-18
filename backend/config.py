from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    FRONTEND_FILE = "Bone detection frontend · HTML.html"
    MODEL_PATHS = [
        BASE_DIR / "knee_model.keras",
        BASE_DIR / "model.h5",
    ]
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_DIR = DATA_DIR / "uploads"
    REPORT_DIR = DATA_DIR / "reports"
    IMG_SIZE = (224, 224)
    CLASSES = ["Normal", "Osteopenia", "Osteoporosis"]
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    DIAGNOSIS_INFO = {
        "Normal": {
            "title": "Healthy bone density",
            "description": (
                "Bone density appears within the normal range. No significant "
                "degenerative changes were detected by the model."
            ),
            "recommendation": (
                "Maintain bone health with regular exercise, balanced nutrition, "
                "and routine follow-up with a qualified clinician."
            ),
            "severity": "normal",
        },
        "Osteopenia": {
            "title": "Mild bone loss",
            "description": (
                "The model detected reduced bone mineral density that may be "
                "consistent with osteopenia."
            ),
            "recommendation": (
                "Discuss the result with a physician. Lifestyle changes, vitamin D, "
                "calcium intake, and additional clinical assessment may be needed."
            ),
            "severity": "warning",
        },
        "Osteoporosis": {
            "title": "Significant bone loss",
            "description": (
                "The model detected substantial bone density loss that may be "
                "consistent with osteoporosis."
            ),
            "recommendation": (
                "Seek medical evaluation promptly. A clinician may recommend "
                "diagnostic confirmation, medication, and fall-prevention planning."
            ),
            "severity": "critical",
        },
    }
