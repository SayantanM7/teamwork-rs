# teamwork-rs
This is the Repository for our MLOPs assignment. Members are Anil,Rajesh and Sayantan.
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

