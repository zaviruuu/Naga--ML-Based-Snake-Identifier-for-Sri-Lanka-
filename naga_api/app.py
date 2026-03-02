from flask import Flask, request, jsonify
from PIL import Image
import base64, io
import tensorflow as tf
import json
import os

app = Flask(__name__)

MODEL_PATH = "models/identifier.h5"   # change name if yours differs
LABELS_PATH = "data/labels.json"            # optional if you have labels.json

# Load model once when server starts
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load labels (optional)
labels = None
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

IMG_SIZE = (224, 224)

def preprocess(pil_img: Image.Image):
    img = pil_img.resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0
    arr = tf.expand_dims(arr, axis=0)  # (1,224,224,3)
    return arr

def predict_snake(pil_img: Image.Image):
    x = preprocess(pil_img)
    preds = model.predict(x, verbose=0)[0]

    idx = int(tf.argmax(preds).numpy())
    conf = float(preds[idx])

    if labels:
        name = labels[idx] if isinstance(labels, list) else labels.get(str(idx), str(idx))
    else:
        name = str(idx)

    return name, conf

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/identify", methods=["POST"])
def identify():
    try:
        data = request.get_json(force=True)

        if "image_base64" not in data:
            return jsonify({"error": "No image_base64 provided"}), 400

        image_bytes = base64.b64decode(data["image_base64"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        snake, confidence = predict_snake(image)

        return jsonify({
            "snake_name": snake,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500