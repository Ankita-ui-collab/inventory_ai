
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

    detected_component = class_names[best_class] if best_class < len(class_names) else "Unknown"

    os.remove(image_path)

    return jsonify({
        "component": detected_component,
        "confidence": round(best_conf, 3)
    })

