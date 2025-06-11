import json
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)


def evaluate_model() -> dict[str, Any]:
    """Evaluate the trained model and return metrics.

    Returns:
        Dictionary containing evaluation metrics
    """
    # Load unique classes from the original features file
    classes = pd.read_csv("data/features_iris.csv")["target"].unique().tolist()

    # Load test dataset
    test_dataset = pd.read_csv("data/test.csv")
    y: np.ndarray = test_dataset.loc[:, "target"].values.astype("float32")
    X: np.ndarray = test_dataset.drop("target", axis=1).values

    # Load trained model
    clf = joblib.load("models/model.joblib")

    # Make predictions
    prediction: np.ndarray = clf.predict(X)

    # Calculate metrics
    cm: np.ndarray = confusion_matrix(y, prediction)
    f1: float = f1_score(y_true=y, y_pred=prediction, average="macro")

    return {
        "f1_score": f1,
        "confusion_matrix": {"classes": classes, "matrix": cm.tolist()},
    }


if __name__ == "__main__":
    metrics = evaluate_model()

    # Save metrics as JSON
    with open("data/eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

    MLFLOW_TRACKING_URI = "http://localhost:5001"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("assignment-3-mlflow")

    clf = joblib.load("models/model.joblib")
    train_dataset = pd.read_csv("data/train.csv")

    X: np.ndarray = train_dataset.drop("target", axis=1).values
    y: np.ndarray = train_dataset.loc[:, "target"].values.astype("float32")

    preds = clf.predict(X)

    me = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)

    with mlflow.start_run() as run:
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_params({"C": 0.01, "solver": "lbfgs", "max_iter": 100})

        mlflow.log_artifact("data/train.csv")
        mlflow.log_artifact("models/model.joblib")

        mlflow.set_tag("logistic-regression", "Logistic Regression")

        with mlflow.start_run(run_name="training", nested=True) as child_run:
            mlflow.log_metric("mean_squared_error", me)
            mlflow.log_metric("mean_absolute_error", mae)

        with mlflow.start_run(run_name="evaluation", nested=True) as child_run:
            mlflow.log_metric("f1_score", metrics["f1_score"])
            mlflow.log_dict(metrics["confusion_matrix"], "confusion_matrix.json")

        signature = mlflow.models.infer_signature(X, clf.predict(X))
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            signature=signature,
            input_example=X,
            registered_model_name="logistic-regression",
            artifact_path="models",
        )
