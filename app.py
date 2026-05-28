
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

    # Convert shape (65,8400) -> (8400,65)
    predictions = predictions.T

    boxes = []
    confidences = []
    class_ids = []

    all_scores = []

    # Collect confidence scores
    for row in predictions:

        class_scores = row[4:]

        class_id = np.argmax(class_scores)

        confidence = class_scores[class_id]

        all_scores.append(float(confidence))

    # Dynamic threshold
    dynamic_threshold = max(np.mean(all_scores) * 1.5, 0.15)

    # Detection filtering
    for row in predictions:

        class_scores = row[4:]

        class_id = np.argmax(class_scores)

        confidence = class_scores[class_id]

        if confidence < dynamic_threshold:
            continue

        x, y, w, h = row[0:4]

        left = int((x - w / 2) * original_w / 640)
        top = int((y - h / 2) * original_h / 640)

        width_box = int(w * original_w / 640)
        height_box = int(h * original_h / 640)

        boxes.append([left, top, width_box, height_box])

        confidences.append(float(confidence))

        class_ids.append(class_id)

    # Apply NMS
    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        score_threshold=dynamic_threshold,
        nms_threshold=0.45
    )

    results = []

    if len(indexes) > 0:

        for i in indexes.flatten():

            detected_class = class_names[class_ids[i]]

            conf = confidences[i]

            results.append({
                "component": detected_class,
                "confidence": round(conf, 3)
            })

    # Fallback prediction if no box survives
    if len(results) == 0:

        best_idx = np.argmax(all_scores)

        best_row = predictions[best_idx]

        best_scores = best_row[4:]

        best_class = np.argmax(best_scores)

        best_conf = np.max(best_scores)

        results.append({
            "component": class_names[best_class],
            "confidence": round(float(best_conf), 3),
            "fallback_prediction": True
        })

    # Delete temp image
    if os.path.exists(image_path):
        os.remove(image_path)

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



