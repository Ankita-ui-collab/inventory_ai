
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load ONNX model
session = ort.InferenceSession("best.onnx")

# Dataset class names
class_names = [
    '1-5-Volt-Battery',
    '3-3-Volt-Battery',
    '7-Segment-Display',
    '9-Volt-Battery',
    'Arduino-Mega',
    'Arduino-Nano',
    'Arduino-Uno',
    'BJT-Transistor',
    'Bluetooth-Module',
    'Breadboard',
    'Bridge-Rectifier',
    'Buck-Converter',
    'Buzzer',
    'Capacitor-10mf',
    'Capacitor-470mf',
    'DC-Motor',
    'Diode',
    'ESP32',
    'ESP32-CAM',
    'FT-232-USB-Serial-Module',
    'Film-Capacitor',
    'Fuse',
    'Fuse-Base',
    'GSM-Module',
    'Gas-Sensor',
    'Heat-Sink',
    'High-Voltage-Ceramic-Capacitor',
    'Humidity-Sensor',
    'IC-Base-14-Pin',
    'IC-Base-28-Pin',
    'IC-Chip',
    'IGBT',
    'IR-Sensor',
    'Inductor',
    'Keypad',
    'LCD-Display',
    'LDR-Sensor',
    'LED-Light',
    'Low-Voltage-Ceramic-Capacitor',
    'MLC-Capacitor',
    'MOSFET',
    'Motion-Sensor',
    'Motor-Driver',
    'NTC-Thermistor',
    'OLED-Display',
    'Pin-Header',
    'Push-Switch',
    'RFID-Scanner',
    'Raindrops-Module',
    'Relay-Module',
    'Resistor',
    'Rocker-Switch',
    'Servo-Motor',
    'Soil-Moisture-Sensor',
    'Sonar-Sensor',
    'TCRT5000',
    'Tact-Switch',
    'Taper-Potentiometer',
    'Trimmer-Potentiometer',
    'Water-Sensor',
    'Zener-Diode'
]


@app.route('/')
def home():
    return "YOLOv8 ONNX Detection Server Running"


@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    image = request.files['image']

    image_path = "temp.jpg"

    image.save(image_path)

    # Read image
    img = cv2.imread(image_path)

    if img is None:
        return jsonify({"error": "Invalid image"})

    original_h, original_w = img.shape[:2]

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

    predictions = outputs[0][0]

    # Convert (65,8400) -> (8400,65)
    predictions = predictions.T

    results = []

    # Collect all predictions
    all_predictions = []

    for row in predictions:

        class_scores = row[4:]

        class_id = np.argmax(class_scores)

        confidence = float(class_scores[class_id])

        # Skip extremely weak detections
        if confidence < 0.10:
            continue

        detected_class = class_names[class_id]

        all_predictions.append({
            "component": detected_class,
            "confidence": confidence
        })

    # Sort by confidence
    all_predictions = sorted(
        all_predictions,
        key=lambda x: x["confidence"],
        reverse=True
    )

    # Remove duplicates
    seen = set()

    top_predictions = []

    for pred in all_predictions:

        component = pred["component"]

        if component not in seen:

            seen.add(component)

            top_predictions.append({
                "component": component,
                "confidence": round(pred["confidence"], 3)
            })

        # Keep top 3 only
        if len(top_predictions) == 3:
            break

    # Fallback if nothing detected
    if len(top_predictions) == 0:

        best_row = predictions[0]

        best_scores = best_row[4:]

        best_class = np.argmax(best_scores)

        best_conf = np.max(best_scores)

        top_predictions.append({
            "component": class_names[best_class],
            "confidence": round(float(best_conf), 3),
            "fallback_prediction": True
        })

    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)

    return jsonify(top_predictions)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

