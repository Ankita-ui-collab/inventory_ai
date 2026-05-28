```python
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load ONNX model
session = ort.InferenceSession("best.onnx")

@app.route('/')

def home():
    return "ONNX AI Server Running"

@app.route('/predict', methods=['POST'])

def predict():

    # Check image upload
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    image = request.files['image']

    image_path = "temp.jpg"

    image.save(image_path)

    # Read image
    img = cv2.imread(image_path)

    # Resize image
    img = cv2.resize(img, (320, 320))

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Change HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Run inference
    outputs = session.run(None, {"images": img})

    os.remove(image_path)

    return jsonify({
        "message": "Inference completed successfully",
        "output_shape": str(np.array(outputs[0]).shape)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

