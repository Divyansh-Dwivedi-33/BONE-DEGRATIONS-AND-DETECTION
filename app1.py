from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
model = load_model("model.h5")  # your trained model

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    name = request.form.get("name")
    age = request.form.get("age")
    sex = request.form.get("sex")
    weight = request.form.get("weight")

    img = Image.open(file).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    result = "Degenerated" if prediction[0][0] > 0.5 else "Healthy"

    return jsonify({
        "name": name,
        "age": age,
        "sex": sex,
        "weight": weight,
        "result": result
    })

if __name__ == "__main__":
    app.run(debug=True)
