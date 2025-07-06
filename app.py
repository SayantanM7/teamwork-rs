from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import base64
import io
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
MODEL_PATH = "model.joblib"
BEST_PARAMS = {}

# Load model if exists
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.route('/train', methods=['POST'])
def train():
    global model, BEST_PARAMS

    try:
        # Load and preprocess data
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float32) / 255.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        mlflow.set_tracking_uri("http://127.0.0.1:5002/")
        mlflow.set_experiment("MLops_digit_recognizer")
        mlflow.sklearn.autolog()


        with mlflow.start_run():
            rf = RandomForestClassifier(random_state=42)
            lr = LogisticRegression(max_iter=1000, random_state=42)

            ensemble = VotingClassifier(estimators=[
                ('rf', rf),
                ('lr', lr)
            ], voting='soft')

            param_grid = {
                'rf__n_estimators': [50],
                'lr__C': [0.01, 0.1, 1.0, 10.0]
            }

            grid_search = GridSearchCV(ensemble, param_grid=param_grid, cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.best_estimator_.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt")

            joblib.dump(grid_search.best_estimator_, MODEL_PATH)
            mlflow.log_artifact(MODEL_PATH)

            model = grid_search.best_estimator_
            BEST_PARAMS = grid_search.best_params_

        return jsonify({
            "message": "Model trained successfully",
            "accuracy": acc,
            "best_params": BEST_PARAMS
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        return jsonify({"error": "Model not trained yet."}), 400

    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode and preprocess image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('L')  # grayscale
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = image_array.reshape(1, -1)

        prediction = model.predict(image_array)[0]
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/best_model_params', methods=['GET'])
def best_model_params():
    global BEST_PARAMS

    if not BEST_PARAMS:
        return jsonify({"message": "Model not trained or parameters not available"}), 400

    return jsonify({"best_params": BEST_PARAMS})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)