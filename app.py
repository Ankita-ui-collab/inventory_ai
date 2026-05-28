from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

model = YOLO("best.pt")

@app.route('/')

def home():
    return "AI Inventory Server Running"

@app.route('/predict', methods=['POST'])

def predict():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    image = request.files['image']

    image_path = "temp.jpg"

    image.save(image_path)

    results = model(image_path)

    detections = []

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            confidence = float(box.conf[0])

            label = model.names[cls]

            detections.append({
                "component": label,
                "confidence": confidence
            })

    os.remove(image_path)

    return jsonify(detections)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
