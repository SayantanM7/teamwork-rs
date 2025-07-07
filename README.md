# teamwork-rs
This is the Repository for our MLOPs assignment. Members are Anil ,Rajesh and Sayantan.
This project builds and deploys a machine learning model to recognize handwritten digits using the MNIST dataset. The model is trained, evaluated, and saved using hyperparameter tuning. 


Dataset: MNIST (from sklearn), 1,797 grayscale 8×8 images of digits. Preprocessed by flattening to 64 features. No normalization or standardization applied (tree-based models handle raw inputs well).

Ensemble Design: Soft VotingClassifier with 2 base models:

LogisticRegression (solver='lbfgs', max_iter=1000)

RandomForestClassifier (n_estimators=50)

Search Space (GridSearchCV, 3-fold):

    lr__C: [0.01, 0.1, 1.0, 10.0]

    rf__n_estimators: [50]

Metrics: Accuracy, Precision, Recall, F1 Score (all weighted), Classification Report. Evaluated on a held-out 20% test set.

Artifacts (via MLflow):

    Trained model: model.joblib and model.pkl

    Classification report: classification_report.txt

    All runs and metrics logged using mlflow.log_*() functions

1. A Dockerized Flask API for MNIST digit recognition with training and prediction endpoints.

 Quick Start

### Pull the Docker Image

 docker pull sayantanm7/teamwork-rsa:v2 



2. Run the container

docker run -p 5000:5000 sayantanm7/teamwork-rsa:v2 

3. API Endpoints
Endpoint	        Method	Description
/train	            POST	Train the model
/best_model_params	GET	    Get best model parameters
/predict	        POST	Predict digit from image


Usage Examples
1. Train the Model
Request:


POST /train
Response:

json
{
    "accuracy": 0.9451428571428572,
    "best_params": {
        "lr__C": 0.1,
        "rf__n_estimators": 50
    },
    "message": "Model trained successfully"
}
2. Get Best Parameters
Request:


GET /best_model_params
Response:

json
{
    "best_params": {
        "lr__C": 0.1,
        "rf__n_estimators": 50
    }
}
3. Predict Digit
Request:


POST /predict
{
    "image": "base64_encoded_image"
}
Response:

json
{
    "prediction": 7
}




🖼️ Image Requirements
Format: 28×28 pixels (grayscale)

Colors: White digit on black background

Encoding: Base64

Python Code to Convert Image
python
from PIL import Image, ImageOps
import base64
import io

def image_to_base64(image_path):
    image = Image.open(image_path).convert("L")
    image = ImageOps.invert(image)  # Ensure black background
    image = image.resize((28, 28))
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# Usage:
b64_img = image_to_base64("your_image.png")
print(b64_img)  # Use this in your API request


Example Base64 Image (Digit "7")



json
{
    "image": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABBklEQVR4nGP8z4AbMOGRo5UkCyr379dfPBzYdX65syDJfu5vhMB/GHh/c2+FFgMDg/tHuBDE2L+Pbj84f+nGOxFHsX1IGiGSlyovPWcQkkw1Nnp3+R+65Osb0s7qxqL6rAznPmozo0labOYQ4GNnYGD4sf21BgvDf0ZkST5dBgaG/wyMDB8f83iwwYMbySuMDP8ZbjyRl2fE7k9GhmN3PAVwhdCHqywu3Axwrahhe/sKDy8SF0nyPyPDlVsq/NglGRh+3f0eJ49L8v19Nlk2XJLPzqhKMGCXZGR4dstFC4ckA8NvBhUulNT4HwH+3Z1+/T8yYERS+Z+RARWghNB/iNUIV5Cd4gHvI3N7sUOzNwAAAABJRU5ErkJggg=="
}

