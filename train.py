from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
def main():
    logger.info("Loading dataset...")
    data = pd.read_csv(DATA_PATH)

    logger.info("Splitting dataset...")

    X = data.drop(columns="HeartDisease")
    y = data["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_features = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "Oldpeak",
    ]

    categorical_features = [
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    logger.info("Training Logistic Regression model...")

    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model...")

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    logger.info(f"Accuracy : {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall   : {recall:.4f}")
    logger.info(f"F1 Score : {f1:.4f}")
    logger.info(f"ROC AUC  : {roc_auc:.4f}")

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }
    logger.info("\nConfusion Matrix\n%s", confusion_matrix(y_test, y_pred))

    logger.info(
        "\nClassification Report\n%s",
        classification_report(y_test, y_pred)
    )

    logger.info("Saving model...")

    MODEL_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Model and metrics saved successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Training failed: {e}")