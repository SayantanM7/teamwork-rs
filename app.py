from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image, ImageOps
import io
import base64

app = Flask(__name__)

# Load trained ensemble model
model = joblib.load("model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 string
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('L')  # Grayscale

        # Resize to 28x28, invert if background is white
        image = ImageOps.invert(image)
        image = image.resize((28, 28))

        # Convert to array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = image_array.reshape(1, -1)  # Flatten

        # Predict
        prediction = model.predict(image_array)[0]

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
