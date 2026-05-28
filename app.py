
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load ONNX model
session = ort.InferenceSession("best.onnx")


@app.route('/predict', methods=['POST'])

def predict():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    image = request.files['image']

    image_path = "temp.jpg"

    image.save(image_path)

    # Read image
    img = cv2.imread(image_path)

    # Resize
    img_resized = cv2.resize(img, (640, 640))

    # Normalize
    img_input = img_resized.astype(np.float32) / 255.0

    # HWC -> CHW
    img_input = np.transpose(img_input, (2, 0, 1))

    # Add batch dimension
    img_input = np.expand_dims(img_input, axis=0)

    # Run inference
    outputs = session.run(None, {"images": img_input})

    output = outputs[0][0]

    # Get highest confidence detection
    scores = output[4:, :]

    class_ids = np.argmax(scores, axis=0)

    confidences = np.max(scores, axis=0)

    best_idx = np.argmax(confidences)

    best_class = int(class_ids[best_idx])

    best_conf = float(confidences[best_idx])

    # Replace with your dataset class names
    class_names = [
        "Arduino",
        "Servo Motor",
        "Relay",
        "ESP32",
        "DHT11"
    ]

    detected_component = class_names[best_class] if best_class < len(class_names) else "Unknown"

    os.remove(image_path)

    return jsonify({
        "component": detected_component,
        "confidence": round(best_conf, 3)
    })

