from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .data_generator import generate_synthetic_data, save_dataset
from .preprocessing import build_preprocessor, get_feature_targets, train_test_split_all


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_all_models(df: pd.DataFrame | None = None, random_state: int = 42) -> dict:
    if df is None:
        df = generate_synthetic_data(random_state=random_state)
        save_dataset(df, DATA_DIR / "synthetic_agri_data.csv")

    X, targets = get_feature_targets(df)
    preprocessor = build_preprocessor(df)

    (
        X_train,
        X_test,
        y_yield_train,
        y_yield_test,
        y_health_train,
        y_health_test,

        
        y_suit_train,
        y_suit_test,
        y_prod_train,
        y_prod_test,
    ) = train_test_split_all(df)

    lin_reg = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])
    lin_reg.fit(X_train, y_yield_train)
    y_pred = lin_reg.predict(X_test)
    # Compute RMSE manually for compatibility with older scikit-learn versions
    mse = mean_squared_error(y_yield_test, y_pred)
    rmse = float(np.sqrt(mse))
    lin_metrics = {
        "R2": r2_score(y_yield_test, y_pred),
        "MAE": mean_absolute_error(y_yield_test, y_pred),
        "RMSE": rmse,
    }

    log_reg = Pipeline([
        ("preprocess", preprocessor),
        (
            "model",
            LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=None),
        ),
    ])
    log_reg.fit(X_train, y_health_train)
    y_health_pred = log_reg.predict(X_test)
    cm = confusion_matrix(y_health_test, y_health_pred, labels=["Healthy", "Unhealthy"])
    log_metrics = {
        "Accuracy": accuracy_score(y_health_test, y_health_pred),
        "Precision_Healthy": cm[0, 0] / max(cm[0, 0] + cm[1, 0], 1),
        "Recall_Healthy": cm[0, 0] / max(cm[0, 0] + cm[0, 1], 1),
        "Confusion_Matrix": cm,
    }

    knn = Pipeline([
        ("preprocess", preprocessor),
        ("model", KNeighborsClassifier(n_neighbors=7)),
    ])
    knn.fit(X_train, y_suit_train)
    y_suit_pred = knn.predict(X_test)
    knn_metrics = {
        "Accuracy": accuracy_score(y_suit_test, y_suit_pred),
        "Report": classification_report(y_suit_test, y_suit_pred, output_dict=False),
    }

    pre_X = preprocessor.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(pre_X)

    joblib.dump(lin_reg, MODELS_DIR / "linear_regression_yield.pkl")
    joblib.dump(log_reg, MODELS_DIR / "logistic_regression_health.pkl")
    joblib.dump(knn, MODELS_DIR / "knn_suitability.pkl")
    joblib.dump({"model": kmeans, "preprocessor": preprocessor}, MODELS_DIR / "kmeans_productivity.pkl")

    metrics = {
        "linear_regression": lin_metrics,
        "logistic_regression": log_metrics,
        "knn": knn_metrics,
    }

    return {
        "metrics": metrics,
        "clusters": clusters,
        "preprocessed_X": pre_X,
        "raw_X": X,
        "targets": targets,
    }


if __name__ == "__main__":
    df = generate_synthetic_data()
    info = train_all_models(df)
    print("Training complete.")
    print("Linear Regression:", info["metrics"]["linear_regression"]) 
