import numpy as np
import pandas as pd
from pathlib import Path


SOIL_TYPES = ["Clay", "Sandy", "Loamy"]
SEASONS = ["Kharif", "Rabi", "Summer"]


def generate_synthetic_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic agricultural dataset with correlated features."""
    rng = np.random.default_rng(random_state)

    soil_type = rng.choice(SOIL_TYPES, size=n_samples, p=[0.3, 0.3, 0.4])
    season = rng.choice(SEASONS, size=n_samples, p=[0.4, 0.3, 0.3])

    base_rainfall = np.select(
        [season == "Kharif", season == "Rabi", season == "Summer"],
        [rng.normal(1200, 200, n_samples), rng.normal(600, 150, n_samples), rng.normal(400, 120, n_samples)],
    )
    rainfall = np.clip(base_rainfall + rng.normal(0, 80, n_samples), 200, 2000)

    base_temp = np.select(
        [season == "Kharif", season == "Rabi", season == "Summer"],
        [rng.normal(28, 3, n_samples), rng.normal(20, 3, n_samples), rng.normal(35, 3, n_samples)],
    )
    temperature = np.clip(base_temp + rng.normal(0, 2, n_samples), 10, 45)

    humidity = np.clip(
        90 - (temperature - 20) * 1.2 + rng.normal(0, 5, n_samples),
        30,
        95,
    )

    fertilizer_usage = np.clip(
        np.select(
            [soil_type == "Clay", soil_type == "Loamy", soil_type == "Sandy"],
            [rng.normal(220, 60, n_samples), rng.normal(200, 50, n_samples), rng.normal(260, 70, n_samples)],
        ),
        0,
        500,
    )

    pesticide_usage = np.clip(
        40 + (humidity - 60) * 0.3 + rng.normal(0, 10, n_samples),
        0,
        100,
    )

    irrigation_hours = np.clip(
        np.select(
            [soil_type == "Clay", soil_type == "Loamy", soil_type == "Sandy"],
            [rng.normal(4, 1.2, n_samples), rng.normal(5, 1.5, n_samples), rng.normal(7, 1.8, n_samples)],
        ),
        0,
        15,
    )

    prev_yield_base = np.select(
        [soil_type == "Clay", soil_type == "Loamy", soil_type == "Sandy"],
        [rng.normal(3.2, 0.4, n_samples), rng.normal(3.8, 0.5, n_samples), rng.normal(2.8, 0.4, n_samples)],
    )
    prev_yield = np.clip(prev_yield_base + (rainfall - 800) / 2000 + rng.normal(0, 0.2, n_samples), 1.0, 6.0)

    optimal_rainfall = 900
    optimal_temp = 26
    optimal_humidity = 70

    yield_mean = (
        1.5
        + 0.003 * np.clip(rainfall - optimal_rainfall, -400, 400)
        - 0.02 * np.abs(temperature - optimal_temp)
        + 0.01 * fertilizer_usage
        - 0.005 * pesticide_usage
        + 0.08 * irrigation_hours
        + 0.6 * prev_yield
    )

    crop_yield = np.clip(yield_mean + rng.normal(0, 0.5, n_samples), 1.0, 8.0)

    health_score = (
        0.3 * (humidity / 100)
        + 0.3 * (1 - np.abs(temperature - optimal_temp) / 20)
        + 0.2 * (1 - pesticide_usage / 120)
        + 0.2 * (fertilizer_usage / 400)
        + rng.normal(0, 0.05, n_samples)
    )
    health_score = np.clip(health_score, 0, 1)
    crop_health = np.where(health_score > 0.55, "Healthy", "Unhealthy")

    suitability_raw = (
        0.4 * (crop_yield / 8)
        + 0.2 * (1 - np.abs(temperature - optimal_temp) / 20)
        + 0.2 * (rainfall / 2000)
        + 0.2 * (fertilizer_usage / 500)
    )
    suitability_score = np.clip(suitability_raw + rng.normal(0, 0.05, n_samples), 0, 1)
    suitability_category = pd.cut(
        suitability_score,
        bins=[-0.01, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
    )

    prod_score = crop_yield + 0.5 * prev_yield
    prod_labels = pd.qcut(prod_score, q=3, labels=["Low Productivity", "Moderate Productivity", "High Productivity"])

    df = pd.DataFrame(
        {
            "Soil_Type": soil_type,
            "Rainfall": rainfall.round(2),
            "Temperature": temperature.round(2),
            "Humidity": humidity.round(2),
            "Fertilizer_Usage": fertilizer_usage.round(2),
            "Pesticide_Usage": pesticide_usage.round(2),
            "Irrigation_Hours": irrigation_hours.round(2),
            "Previous_Yield": prev_yield.round(2),
            "Season": season,
            "Crop_Yield": crop_yield.round(2),
            "Crop_Health": crop_health,
            "Suitability": suitability_category.astype(str),
            "Productivity_Zone": prod_labels.astype(str),
        }
    )

    return df


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    data = generate_synthetic_data()
    csv_path = save_dataset(data, Path(__file__).resolve().parents[1] / "data" / "synthetic_agri_data.csv")
    print(f"Dataset saved to {csv_path}")
