# Smart Crop Yield Prediction & Advisory System

End-to-end data science web project built with Python, scikit-learn, and Streamlit.  
It generates realistic synthetic agricultural data, trains multiple ML models, and exposes
predictions and advisory insights via a multi-page web UI.

## Features

- Synthetic dataset generator (\>= 1000 records) with realistic correlations.
- ML tasks:
  - **Linear Regression** – Crop Yield prediction (R2, MAE, RMSE).
  - **Logistic Regression** – Crop Health (Healthy/Unhealthy).
  - **KNN Classification** – Crop Suitability (High/Medium/Low).
  - **K-Means Clustering** – Farm productivity segmentation.
- Preprocessing with categorical encoding and feature scaling.
- Models saved as `.pkl` files in `models/`.
- Multi-page Streamlit app with:
  - Home (overview & architecture).
  - Data (generate / upload / download / preview).
  - Prediction (forms + advisory + rainfall simulation + fertilizer recommendation).
  - Visualization (heatmap, yield distribution, clusters, feature importance).

## Project Structure

```bash
smart-crop-system/
├── app.py                 # Streamlit app entrypoint
├── requirements.txt
├── README.md
├── data/                  # Synthetic and uploaded datasets (created at runtime)
├── models/                # Trained model artifacts (.pkl, created at runtime)
└── src/
    ├── data_generator.py  # Synthetic data generation utilities
    ├── preprocessing.py   # Feature extraction & preprocessing
    ├── train_models.py    # Model training & metric computation
    └── predict.py         # Unified prediction & advisory logic
```

## Setup & Installation

1. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

From the `smart-crop-system` directory, run:

```bash
streamlit run app.py
```

On first launch, the app will:

- Generate a synthetic dataset (if none exists in `data/`).
- Train all ML models and save them under `models/`.

Subsequent runs will reuse existing models unless you click **"Retrain Models"** in the sidebar.

## Notes

- You can upload your own CSV with compatible columns on the **Data** page.
- Cluster visualizations and feature importance rely on trained models; if they are missing,
  simply retrain from the sidebar.
- This code is intended to be hackathon-ready and easy to extend.
