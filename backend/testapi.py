from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, "skin_cancer_model.keras")
label_encoder_path = os.path.join(script_dir, "class_indices.json")

model = load_model(model_path, compile=False)

with open(label_encoder_path, "r") as f:  
    loaded_data = json.load(f)

label_encoder = []

if isinstance(loaded_data, list):
    label_encoder = loaded_data
elif isinstance(loaded_data, dict):
    if all(isinstance(v, int) for v in loaded_data.values()):
        sorted_items = sorted(loaded_data.items(), key=lambda item: item[1])
        label_encoder = [item[0] for item in sorted_items]
    else:
        try:
            sorted_items = sorted(loaded_data.items(), key=lambda item: int(item[0]))
            label_encoder = [item[1] for item in sorted_items]
        except ValueError:
            label_encoder = list(loaded_data.values())


disease_info = {
    "akiec": "Actinic Keratoses — precancerous lesions caused by sun exposure.",
    "bcc": "Basal Cell Carcinoma — slow-growing cancer, rarely spreads.",
    "bkl": "Benign Keratosis-like lesions — non-cancerous skin growths.",
    "df": "Dermatofibroma — benign fibrous skin nodule.",
    "mel": "Melanoma — dangerous and aggressive skin cancer.",
    "nv": "Melanocytic Nevi — common mole, usually benign.",
    "vasc": "Vascular Lesion — benign blood vessel growths."
}

@app.route('/', methods=['GET'])
def home():
    """
    Main page to test if the server is running.
    Accessing http://127.0.0.1:5000/ in a browser will trigger this.
    """
    return "Skin Cancer Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    """
    if 'image' not in request.files:
        return jsonify({"error": "Image not provided with key 'image'."}), 400

    file = request.files['image']
    
    try:
        IMG_SIZE = (300, 300)  
        img = Image.open(file.stream).convert("RGB").resize(IMG_SIZE)
        
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  

        preds = model.predict(img_array)
        pred_class_index = int(np.argmax(preds))
        confidence = float(np.max(preds)) 

        class_name = label_encoder[pred_class_index]

        return jsonify({
            "prediction": class_name,
            "confidence": confidence * 100,  
            "description": disease_info.get(class_name, "No description available for this disease."),
            "all_probabilities": {label: float(prob) for label, prob in zip(label_encoder, preds[0])}
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)