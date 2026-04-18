from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf
from PIL import Image


class ModelService:
    def __init__(self, model_paths, image_size, class_names, diagnosis_info):
        self._model_paths = [Path(path) for path in model_paths]
        self._image_size = image_size
        self._class_names = class_names
        self._diagnosis_info = diagnosis_info
        self._model = None
        self._lock = Lock()

    def health_status(self) -> dict:
        model_path = self._resolve_model_path()
        return {
            "status": "ok",
            "model_loaded": self._model is not None,
            "model_path": str(model_path) if model_path else None,
            "classes": self._class_names,
        }

    def predict(self, image_path: str) -> dict:
        model = self._get_model()
        image_batch = self._preprocess_image(image_path)
        raw_probs = model.predict(image_batch, verbose=0)[0]

        probabilities = {
            label: float(probability)
            for label, probability in zip(self._class_names, raw_probs.tolist())
        }
        predicted_label = max(probabilities, key=probabilities.get)
        diagnosis = self._diagnosis_info[predicted_label]

        return {
            "label": predicted_label,
            "confidence": probabilities[predicted_label],
            "probabilities": probabilities,
            "probs": [probabilities[label] for label in self._class_names],
            "title": diagnosis["title"],
            "description": diagnosis["description"],
            "recommendation": diagnosis["recommendation"],
            "severity": diagnosis["severity"],
        }

    def _get_model(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                model_path = self._resolve_model_path()
                if model_path is None:
                    expected = ", ".join(str(path) for path in self._model_paths)
                    raise FileNotFoundError(
                        "No trained model found. Expected one of: "
                        f"{expected}. Run training.py first."
                    )
                self._model = tf.keras.models.load_model(model_path)

        return self._model

    def _resolve_model_path(self):
        for path in self._model_paths:
            if path.exists():
                return path
        return None

    def _preprocess_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB").resize(self._image_size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0)
