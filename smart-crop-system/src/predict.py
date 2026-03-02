from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


@dataclass
class PredictionResult:
    predicted_yield: float
    health_status: str
    suitability: str
    productivity_cluster: str
    advisory: str
    fertilizer_recommendation: float


PRODUCTIVITY_LABELS = {
    0: "Low Productivity",
    1: "Moderate Productivity",
    2: "High Productivity",
}


def load_models() -> Dict[str, Any]:
    lin_reg = joblib.load(MODELS_DIR / "linear_regression_yield.pkl")
    log_reg = joblib.load(MODELS_DIR / "logistic_regression_health.pkl")
    knn = joblib.load(MODELS_DIR / "knn_suitability.pkl")
    km_bundle = joblib.load(MODELS_DIR / "kmeans_productivity.pkl")
    return {
        "linear_reg": lin_reg,
        "log_reg": log_reg,
        "knn": knn,
        "kmeans": km_bundle["model"],
        "preprocessor": km_bundle["preprocessor"],
    }


def _build_input_df(payload: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([payload])[
        [
            "Soil_Type",
            "Rainfall",
            "Temperature",
            "Humidity",
            "Fertilizer_Usage",
            "Pesticide_Usage",
            "Irrigation_Hours",
            "Previous_Yield",
            "Season",
        ]
    ]


def advisory_logic(pred_yield: float, health: str, suitability: str, rainfall: float, fertilizer: float) -> str:
    advice = []
    if pred_yield < 3:
        advice.append("Low projected yield. Consider improving soil fertility and irrigation schedule.")
    elif pred_yield < 5:
        advice.append("Moderate yield expected. Fine-tune nutrient management and monitor pests.")
    else:
        advice.append("High yield projected. Maintain current best practices and timely operations.")

    if health == "Unhealthy":
        advice.append("Signs of crop stress. Inspect for pests, diseases, or nutrient deficiencies.")

    if suitability == "Low":
        advice.append("Field conditions suboptimal. Re-evaluate crop choice or adjust planting window.")

    if rainfall < 600:
        advice.append("Low rainfall scenario. Prioritize drought-tolerant varieties and efficient irrigation.")
    elif rainfall > 1400:
        advice.append("High rainfall. Ensure proper drainage to avoid waterlogging and root diseases.")

    if fertilizer < 120:
        advice.append("Fertilizer usage is low. Consider soil testing and balanced NPK application.")
    # Join all advisory messages into a single paragraph
    return " ".join(advice)


def fertilizer_recommendation(soil: str, prev_yield: float, rainfall: float) -> float:
    base = 200.0
    if soil == "Sandy":
        base += 40
    elif soil == "Clay":
        base -= 20

    if prev_yield > 4.5:
        base -= 30
    elif prev_yield < 2.5:
        base += 30

    if rainfall < 600:
        base -= 20
    elif rainfall > 1200:
        base += 10

    return float(max(80.0, min(320.0, base)))


def predict_all(models: Dict[str, Any], features: Dict[str, Any]) -> PredictionResult:
    df = _build_input_df(features)

    lin_reg = models["linear_reg"]
    log_reg = models["log_reg"]
    knn = models["knn"]
    kmeans = models["kmeans"]
    preprocessor = models["preprocessor"]

    pred_yield = float(lin_reg.predict(df)[0])
    health = str(log_reg.predict(df)[0])
    suit = str(knn.predict(df)[0])

    pre_x = preprocessor.transform(df)
    cluster_id = int(kmeans.predict(pre_x)[0])
    prod_zone = PRODUCTIVITY_LABELS.get(cluster_id, "Moderate Productivity")

    fert_rec = fertilizer_recommendation(
        soil=features["Soil_Type"],
        prev_yield=float(features["Previous_Yield"]),
        rainfall=float(features["Rainfall"]),
    )

    advice = advisory_logic(
        pred_yield,
        health,
        suit,
        rainfall=float(features["Rainfall"]),
        fertilizer=float(features["Fertilizer_Usage"]),
    )

    return PredictionResult(
        predicted_yield=round(pred_yield, 2),
        health_status=health,
        suitability=suit,
        productivity_cluster=prod_zone,
        advisory=advice,
        fertilizer_recommendation=round(fert_rec, 1),
    )
