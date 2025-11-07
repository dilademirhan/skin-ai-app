from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model("skin_cancer_fast.keras")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

disease_info = {
    "akiec": "Actinic Keratoses — precancerous lesions caused by sun exposure.",
    "bcc": "Basal Cell Carcinoma — slow-growing cancer, rarely spreads.",
    "bkl": "Benign Keratosis — non-cancerous skin growths.",
    "df": "Dermatofibroma — benign fibrous skin nodule.",
    "mel": "Melanoma — dangerous and aggressive skin cancer.",
    "nv": "Melanocytic Nevus — common mole, usually benign.",
    "vasc": "Vascular Lesion — benign blood vessel growths."
}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file).convert("RGB").resize((300, 300))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)

    preds = model.predict(img_array)
    pred_class = np.argmax(preds)
    confidence = float(np.max(preds))

    class_name = label_encoder.inverse_transform([pred_class])[0]
    description = disease_info.get(class_name, "")

    return jsonify({
        "prediction": class_name,
        "confidence": confidence,
        "description": description
    })

if __name__ == "__main__":
    app.run(debug=True)
