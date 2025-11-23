from flask import Flask, request, jsonify
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import json
import os
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

script_dir = os.path.dirname(os.path.abspath(__file__))

weights_path = os.path.join(script_dir, "final_best_weights_64.h5") 
label_encoder_path = os.path.join(script_dir, "class_indices.json")

IMG_SIZE = (224, 224)
NUM_CLASSES = 7 

base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=IMG_SIZE + (3,)
)

base_model.trainable = False 

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("Model mimarisi oluşturuldu. Ağırlıklar yükleniyor...")
try:
    model.load_weights(weights_path, by_name=True) 
    print("Model succesfully loaded!")
except Exception as e:
    print(f"Error: Weights could not be loaded! Is the file located at {weights_path}? Error: {e}")
    raise e

with open(label_encoder_path, "r") as f:
    loaded_data = json.load(f)

label_encoder = [item[0] for item in sorted(loaded_data.items(), key=lambda item: item[1])]

if len(label_encoder) != NUM_CLASSES:
    print(f"WARNING: Num classes ({NUM_CLASSES}) does not match the number of classes in JSON ({len(label_encoder)}).")
    # NUM_CLASSES = len(label_encoder) 


disease_info = {
    "akiec": "Actinic Keratoses — precancerous lesions caused by sun exposure. May develop into squamous cell carcinoma. Please consult a dermatologist for monitoring and treatment options.",
    "bcc": "Basal Cell Carcinoma — slow-growing cancer, rarely spreads. Usually appears as a pearly bump. Early treatment is important to prevent tissue damage. Consult a dermatologist for options.",
    "bkl": "Benign Keratosis-like lesions — non-cancerous skin growths. Generally harmless but should be monitored for changes. If you notice any changes in size, color, or shape, please see a dermatologist.",
    "df": "Dermatofibroma — benign fibrous skin nodule. Usually harmless but can be removed if bothersome. Consult a dermatologist for evaluation and removal options.",
    "mel": "Melanoma — dangerous and aggressive skin cancer. Early detection is crucial. If you notice changes in size, shape, or color of a mole, seek immediate medical attention from a dermatologist.",
    "nv": "Melanocytic Nevi — common mole, usually benign. Monitor for changes in size, color, or shape. If changes occur, consult a dermatologist for evaluation.",
    "vasc": "Vascular Lesion — benign blood vessel growths. Usually harmless but should be monitored. If you notice changes or symptoms, please see a dermatologist for evaluation."
}


@app.route('/', methods=['GET'])
def home():
    return "Skin Cancer Prediction API is running (Functional API - MobileNetV2)!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Image not provided with key 'image'."}), 400

    file = request.files['image']
    
    try:
        img = Image.open(file.stream).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img)
        
        processed_img = preprocess_input(img_array) 
        
        processed_img = np.expand_dims(processed_img, axis=0) 

        
        preds = model.predict(processed_img)
        pred_class_index = int(np.argmax(preds))
        confidence = float(np.max(preds))
        
        original_class_name = str(label_encoder[pred_class_index])

        CONFIDENCE_THRESHOLD = 0.10 

        if confidence < CONFIDENCE_THRESHOLD:
            final_prediction = "Undetected"
            final_description = "The accuracy is below the confidence threshold (50%). A lesion couldn't be detected. Please upload a clearer image or consult a dermatologist for accurate diagnosis."
        else:
            final_prediction = original_class_name
            final_description = disease_info.get(final_prediction, "No description available for this disease.")

        return jsonify({
            "prediction": final_prediction,
            "confidence": confidence * 100,
            "description": final_description,
            "all_probabilities": {label: float(prob) for label, prob in zip(label_encoder, preds[0])}
        })

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)