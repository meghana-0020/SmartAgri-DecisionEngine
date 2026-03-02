from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
import time

from src.data_generator import generate_synthetic_data, save_dataset
from src.predict import PredictionResult, load_models, predict_all
from src.train_models import BASE_DIR, DATA_DIR, MODELS_DIR, train_all_models


st.set_page_config(
    page_title="Smart Crop Yield Prediction & Advisory System",
    layout="wide",
)


# Light, professional plotting theme
sns.set_theme(style="whitegrid", palette="crest")


def inject_custom_css() -> None:
    """Inject custom CSS for a light, premium dashboard look."""
    st.markdown(
        """
        <style>
        /* Overall page */
        .main {
            background: radial-gradient(circle at top left, #f8fafc, #eef2ff, #e0f2fe);
            color: #1f2937; /* primary dark slate */
            animation: fadeInPage 0.6s ease-in-out;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f3f4ff, #e6f0ff, #eafaf1) !important;
            border-right: 1px solid #e5e7eb;
            color: #1f2937;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #1f2937;
            font-weight: 700;
        }

        p, span, li, label, .stMarkdown {
            color: #374151;
        }

        /* Metric cards (glassmorphism) */
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.65);
            padding: 0.9rem 1.15rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
            backdrop-filter: blur(14px);
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-6px);
            box-shadow: 0 12px 40px rgba(15, 23, 42, 0.14);
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #60a5fa, #a855f7);
            color: #1f2937;
            border-radius: 999px;
            border: none;
            padding: 0.5rem 1.6rem;
            font-weight: 700;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.12);
            transition: transform 0.18s ease-out, box-shadow 0.18s ease-out, filter 0.18s ease-out;
        }

        .stButton>button:hover {
            filter: brightness(1.04);
            transform: translateY(-2px) scale(1.03);
            box-shadow: 0 10px 26px rgba(129, 140, 248, 0.45);
        }

        /* Cards / containers */
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2.1rem;
        }

        .card {
            background: rgba(255,255,255,0.65);
            border-radius: 20px;
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
            margin-bottom: 1rem;
            backdrop-filter: blur(14px);
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        }

        .card:hover {
            transform: translateY(-6px);
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.16);
        }

        /* Sidebar icon navigation */
        .sidebar-icon {
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            cursor: pointer;
            transition: background 0.15s ease-out, transform 0.15s ease-out, box-shadow 0.15s ease-out;
        }

        .sidebar-icon:hover {
            background: rgba(30, 64, 175, 0.65);
            box-shadow: 0 0 18px rgba(59,130,246,0.8);
            transform: translateY(-1px) scale(1.03);
        }

        .sidebar-icon span {
            font-size: 0.9rem;
        }

        /* Dataframe table styling */
        .stDataFrame, .stTable {
            border-radius: 0.75rem;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.6);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
        }

        .stDataFrame table, .stTable table {
            border-collapse: separate !important;
        }

        .stDataFrame tbody tr:nth-child(even), .stTable tbody tr:nth-child(even) {
            background-color: #f1f5f9;
        }

        .stDataFrame thead tr, .stTable thead tr {
            background-color: #e5f0ff !important;
        }

        /* Result cards */
        .result-card {
            padding: 1rem 1.2rem;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.4);
            box-shadow: 0 10px 30px rgba(59,130,246,0.25);
            background: radial-gradient(circle at top left, rgba(191, 219, 254, 0.7), rgba(240,249,255,0.98));
            backdrop-filter: blur(16px);
            animation: slideUp 0.6s ease-out, fadeIn 0.6s ease-out, scalePop 0.6s ease-out;
            transition: all 0.3s ease-in-out;
        }

        .result-yield { border-left: 3px solid #22c55e; }
        .result-health { border-left: 3px solid #22c55e; }
        .result-suit { border-left: 3px solid #3b82f6; }
        .result-prod { border-left: 3px solid #f97316; }

        /* Plot containers */
        [data-testid="stPlotlyChart"], [data-testid="stVerticalBlock"] .stPlot {
            border-radius: 0.9rem;
            border: 1px solid rgba(148,163,184,0.7);
            box-shadow: 0 10px 30px rgba(15,23,42,0.18);
        }

        /* Footer */
        .app-footer {
            margin-top: 1.5rem;
            text-align: center;
            font-size: 0.8rem;
            color: #6b7280;
        }

        /* Simple background particles */
        .particle-bg {
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                radial-gradient(circle at 15% 15%, rgba(129,140,248,0.10) 0, transparent 55%),
                radial-gradient(circle at 80% 20%, rgba(96,165,250,0.08) 0, transparent 55%),
                radial-gradient(circle at 20% 80%, rgba(125,211,252,0.07) 0, transparent 55%);
            opacity: 0.6;
            z-index: -1;
            animation: floatParticles 22s ease-in-out infinite alternate;
        }

        /* Animations */
        @keyframes fadeInPage {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(12px) scale(0.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes scalePop {
            0% { transform: translateY(12px) scale(0.94); }
            60% { transform: translateY(0) scale(1.03); }
            100% { transform: translateY(0) scale(1); }
        }

        @keyframes floatParticles {
            from { transform: translateY(0px); }
            to { transform: translateY(-18px); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_or_generate_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "synthetic_agri_data.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    df = generate_synthetic_data()
    save_dataset(df, csv_path)
    return df


def ensure_models_trained():
    required_files = [
        MODELS_DIR / "linear_regression_yield.pkl",
        MODELS_DIR / "logistic_regression_health.pkl",
        MODELS_DIR / "knn_suitability.pkl",
        MODELS_DIR / "kmeans_productivity.pkl",
    ]
    if not all(f.exists() for f in required_files):
        with st.spinner("Training models for the first time. Please wait..."):
            info = train_all_models(load_or_generate_data())
            st.session_state["metrics"] = info["metrics"]
    else:
        # Try load metrics from session or recompute quickly from data
        if "metrics" not in st.session_state:
            with st.spinner("Loading model performance metrics..."):
                df = load_or_generate_data()
                info = train_all_models(df)
                st.session_state["metrics"] = info["metrics"]


def home_page():
    df = st.session_state.get("data_df", load_or_generate_data())

    # Hero section
    st.markdown(
        """
        <div class="card" style="margin-bottom:1.5rem;">
            <h1 style="margin-bottom:0.2rem;">🌾 Smart Crop Yield Prediction &amp; Advisory System</h1>
            <p style="margin-top:0.1rem;color:#4b5563;font-weight:600;">AI-powered decision support for modern agriculture.</p>
            <p style="margin-top:0.4rem;font-weight:700;">
                <span style="background:linear-gradient(90deg,#22c55e,#3b82f6);-webkit-background-clip:text;color:transparent;">
                    Predict • Diagnose • Optimize crop performance in seconds.
                </span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # High-level metrics
    total_farms = len(df)
    avg_yield = float(df["Crop_Yield"].mean()) if "Crop_Yield" in df else 0.0
    healthy_pct = (
        float((df["Crop_Health"] == "Healthy").mean() * 100) if "Crop_Health" in df else 0.0
    )
    zone_counts = df["Productivity_Zone"].value_counts(normalize=True) * 100 if "Productivity_Zone" in df else None
    high_zone = float(zone_counts.get("High Productivity", 0.0)) if zone_counts is not None else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🏡 Total Farms", f"{total_farms:,}")
    with m2:
        st.metric("🌾 Avg Yield (t/ha)", f"{avg_yield:.2f}")
    with m3:
        st.metric("💚 Healthy Crops", f"{healthy_pct:.1f}%")
    with m4:
        st.metric("🚀 High Productivity Zones", f"{high_zone:.1f}%")

    st.subheader("Architecture Overview")
    st.markdown(
        """\
        - **Data & Storage:** Synthetic dataset generation + CSV handling (`src/data_generator.py`, `data/`).  
        - **ML Pipeline:** Preprocessing and model training (`src/preprocessing.py`, `src/train_models.py`).  
        - **Inference & Advisory:** Unified prediction + suggestions (`src/predict.py`).  
        - **Web UI:** Streamlit multi-page dashboard (`app.py`).
        """
    )


def data_page():
    st.title("📊 Data Management")
    st.write("Generate, upload, preview, and download the crop dataset.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Generate Synthetic Dataset")
        if st.button("Generate New Synthetic Data", type="primary"):
            df = generate_synthetic_data()
            save_dataset(df, DATA_DIR / "synthetic_agri_data.csv")
            st.session_state["data_df"] = df
            st.success("New synthetic dataset generated and saved.")

    with col2:
        st.subheader("Upload Existing Dataset (CSV)")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state["data_df"] = df
            st.success("Uploaded dataset loaded into session.")

    df = st.session_state.get("data_df", load_or_generate_data())

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Download Dataset")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="crop_dataset.csv",
        mime="text/csv",
    )


def prediction_page():
    st.title("🤖 Smart Prediction & Advisory")

    ensure_models_trained()
    models = load_models()

    df = st.session_state.get("data_df", load_or_generate_data())

    st.subheader("Input Farm Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        soil = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy"], index=2)
        season = st.selectbox("Season", ["Kharif", "Rabi", "Summer"], index=0)
        prev_yield = st.slider("Previous Yield (tons/hectare)", 1.0, 8.0, 3.5, 0.1)

    with col2:
        base_rainfall = st.slider("Base Rainfall (mm)", 200.0, 2000.0, 900.0, 10.0)
        temp = st.slider("Temperature (°C)", 10.0, 45.0, 28.0, 0.5)
        humidity = st.slider("Humidity (%)", 30.0, 95.0, 70.0, 1.0)

    with col3:
        fert_usage = st.slider("Fertilizer Usage (kg)", 0.0, 500.0, 200.0, 5.0)
        pest_usage = st.slider("Pesticide Usage (kg)", 0.0, 100.0, 40.0, 1.0)
        irrigation = st.slider("Irrigation Hours", 0.0, 15.0, 5.0, 0.5)

    st.markdown("**Rainfall Simulation**")
    sim_delta = st.slider("Adjust Rainfall Scenario (mm)", -400.0, 400.0, 0.0, 10.0)
    rainfall_effective = float(np.clip(base_rainfall + sim_delta, 200, 2000))
    st.info(f"Effective rainfall used for prediction: **{rainfall_effective:.1f} mm**")

    if st.button("Run Prediction", type="primary"):
        features = {
            "Soil_Type": soil,
            "Rainfall": rainfall_effective,
            "Temperature": temp,
            "Humidity": humidity,
            "Fertilizer_Usage": fert_usage,
            "Pesticide_Usage": pest_usage,
            "Irrigation_Hours": irrigation,
            "Previous_Yield": prev_yield,
            "Season": season,
        }

        # AI reveal animation sequence
        status_placeholder = st.empty()
        progress_placeholder = st.empty()

        status_placeholder.markdown("**🔍 Analyzing soil conditions...**")
        for i in range(0, 35, 5):
            progress_placeholder.progress(min(i, 100))
            time.sleep(0.08)

        status_placeholder.markdown("**⚙ Optimizing crop parameters...**")
        for i in range(35, 70, 7):
            progress_placeholder.progress(min(i, 100))
            time.sleep(0.08)

        status_placeholder.markdown("**🧠 Generating AI advisory...**")
        for i in range(70, 101, 6):
            progress_placeholder.progress(min(i, 100))
            time.sleep(0.08)

        with st.spinner("Finalizing predictions..."):
            result: PredictionResult = predict_all(models, features)

        # Persist last prediction for advisory page
        st.session_state["last_prediction"] = {
            "features": features,
            "result": result,
        }

        # Clear status elements once done
        status_placeholder.empty()
        progress_placeholder.empty()

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown(
                f"""
                <div class="result-card result-yield">
                    <h4>🌾 Predicted Yield</h4>
                    <p style="font-size:1.4rem;font-weight:700;">{result.predicted_yield:.2f} t/ha</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f"""
                <div class="result-card result-health">
                    <h4>💚 Crop Health</h4>
                    <p style="font-size:1.2rem;font-weight:600;">{result.health_status}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                f"""
                <div class="result-card result-suit">
                    <h4>🌱 Suitability</h4>
                    <p style="font-size:1.2rem;font-weight:600;">{result.suitability}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_d:
            st.markdown(
                f"""
                <div class="result-card result-prod">
                    <h4>🗺 Productivity Zone</h4>
                    <p style="font-size:1.1rem;font-weight:600;">{result.productivity_cluster}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Smart Fertilizer Recommendation")
        st.success(
            f"Recommended Fertilizer Application: **{result.fertilizer_recommendation:.1f} kg** per hectare"
        )

        st.subheader("Crop Advisory Suggestions")
        st.write(result.advisory)

        st.subheader("Model Performance Metrics")
        metrics = st.session_state.get("metrics", {})
        if metrics:
            col1, col2, col3 = st.columns(3)

            lin_m = metrics.get("linear_regression", {})
            log_m = metrics.get("logistic_regression", {})
            knn_m = metrics.get("knn", {})

            with col1:
                st.markdown("**Linear Regression (Yield)**")
                st.write({k: round(v, 3) for k, v in lin_m.items() if k != "Confusion_Matrix"})

            with col2:
                st.markdown("**Logistic Regression (Health)**")
                disp = {k: (round(v, 3) if isinstance(v, (float, int)) else v) for k, v in log_m.items() if k != "Confusion_Matrix"}
                st.write(disp)

            with col3:
                st.markdown("**KNN (Suitability)**")
                st.write({"Accuracy": round(knn_m.get("Accuracy", 0.0), 3)})
        else:
            st.info("Metrics not available yet. They will appear after training.")


def visualization_page():
    st.title("📈 Data & Cluster Visualizations")

    df = st.session_state.get("data_df", load_or_generate_data())

    tab_corr, tab_yield, tab_clusters = st.tabs(["🔗 Correlation", "📊 Yield Distribution", "🧭 Clusters"])

    with tab_corr:
        num_cols = [
            "Rainfall",
            "Temperature",
            "Humidity",
            "Fertilizer_Usage",
            "Pesticide_Usage",
            "Irrigation_Hours",
            "Previous_Yield",
            "Crop_Yield",
        ]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    with tab_yield:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Crop_Yield"], bins=30, kde=True, ax=ax2, color="green")
        ax2.set_xlabel("Crop Yield (tons/hectare)")
        st.pyplot(fig2)

    with tab_clusters:
        ensure_models_trained()
        models = load_models()

        preprocessor = models["preprocessor"]
        kmeans = models["kmeans"]

        feature_df = df[
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
        pre_X = preprocessor.transform(feature_df)
        clusters = kmeans.predict(pre_X)

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        scatter = ax3.scatter(
            df["Rainfall"],
            df["Previous_Yield"],
            c=clusters,
            cmap="viridis",
            alpha=0.7,
        )
        ax3.set_xlabel("Rainfall (mm)")
        ax3.set_ylabel("Previous Yield (t/ha)")
        ax3.set_title("Farm Productivity Clusters")
        legend1 = ax3.legend(*scatter.legend_elements(), title="Cluster")
        ax3.add_artist(legend1)
        st.pyplot(fig3)

    st.subheader("Feature Importance (Linear Model Coefficients)")
    try:
        lin_reg = joblib.load(MODELS_DIR / "linear_regression_yield.pkl")
        preprocessor = lin_reg.named_steps["preprocess"]
        model = lin_reg.named_steps["model"]

        ohe = preprocessor.named_transformers_["cat"]
        num_cols = preprocessor.transformers_[1][2]
        cat_cols = preprocessor.transformers_[0][2]

        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names = cat_feature_names + list(num_cols)

        coefs = model.coef_
        fi_df = pd.DataFrame({"feature": feature_names, "importance": coefs})
        fi_df = fi_df.sort_values("importance", key=abs, ascending=False).head(15)

        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=fi_df, x="importance", y="feature", ax=ax4, palette="viridis")
        ax4.set_title("Top Linear Model Coefficients for Yield Prediction")
        st.pyplot(fig4)
    except Exception as e:  # noqa: F841
        st.info("Feature importance will be available after models are trained.")


def advisory_page():
    st.title("⚙ Smart Advisory System")

    st.markdown("Optimize fertilizer and irrigation with scenario-based guidance.")

    last = st.session_state.get("last_prediction")
    if not last:
        st.info("Run a prediction first on the 'Prediction' page to unlock personalized advisory.")
        return

    features = last["features"]
    base_result: PredictionResult = last["result"]

    col1, col2 = st.columns(2)
    with col1:
        rain_adj = st.slider("🌧 Rainfall scenario (mm)", -400.0, 400.0, 0.0, 10.0)
        fert_adj = st.slider("🧪 Fertilizer adjustment (kg)", -80.0, 80.0, 0.0, 5.0)
        irr_adj = st.slider("💧 Irrigation adjustment (hours)", -4.0, 4.0, 0.0, 0.5)

    with col2:
        st.markdown("**Baseline yield:** {:.2f} t/ha".format(base_result.predicted_yield))
        st.markdown("**Baseline fertilizer:** {:.1f} kg".format(features["Fertilizer_Usage"]))

    models = load_models()

    # Yield change curve vs rainfall
    rain_values = np.linspace(
        max(200, features["Rainfall"] - 300),
        min(2000, features["Rainfall"] + 300),
        25,
    )
    preds = []
    for r in rain_values:
        new_feats = {**features, "Rainfall": float(r)}
        res = predict_all(models, new_feats)
        preds.append(res.predicted_yield)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rain_values, preds, color="#22c55e")
    ax.axvline(features["Rainfall"], color="#f97316", linestyle="--", label="Current")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Predicted Yield (t/ha)")
    ax.legend()
    st.pyplot(fig)

    # Simple qualitative advisory based on adjustments
    st.subheader("Risk Indicator & Advisory")
    adj_rain = features["Rainfall"] + rain_adj
    adj_fert = features["Fertilizer_Usage"] + fert_adj
    adj_irr = features["Irrigation_Hours"] + irr_adj

    risk_score = 0.0
    if adj_rain < 600 or adj_rain > 1400:
        risk_score += 0.4
    if adj_fert < 120 or adj_fert > 320:
        risk_score += 0.3
    if adj_irr < 3 or adj_irr > 10:
        risk_score += 0.3
    risk_score = min(1.0, risk_score)

    st.progress(1 - risk_score, text="Lower bar = higher risk")

    lines = []
    if risk_score <= 0.3:
        lines.append("Current plan is **low risk**. Maintain good agronomic practices.")
    elif risk_score <= 0.6:
        lines.append("Scenario is **moderate risk**. Monitor soil moisture and crop health closely.")
    else:
        lines.append("Scenario is **high risk**. Reconsider irrigation and nutrient schedule.")

    st.write(" ".join(lines))


def main():
    # Apply global custom styling
    inject_custom_css()

    # Background particles
    st.markdown('<div class="particle-bg"></div>', unsafe_allow_html=True)

    # Top bar with title (left) and navigation (right)
    nav_labels = {
        "🏠 Home": "Home",
        "📊 Data": "Data",
        "🔮 Prediction": "Prediction",
        "📈 Visualization": "Visualization",
        "⚙ Smart Advisory": "Advisory",
    }
    with st.container():
        c1, c2 = st.columns([3, 3])
        with c1:
            st.markdown("### 🌾 Smart Crop Dashboard")
        with c2:
            choice = st.radio(
                "",
                list(nav_labels.keys()),
                index=0,
                horizontal=True,
            )
    page = nav_labels[choice]

    # Sidebar only for controls
    st.sidebar.markdown("### Controls")
    if st.sidebar.button("Retrain Models"):
        with st.spinner("Retraining all models..."):
            info = train_all_models(load_or_generate_data())
            st.session_state["metrics"] = info["metrics"]
        st.success("Models retrained successfully.")

    if page == "Home":
        home_page()
    elif page == "Data":
        data_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Visualization":
        visualization_page()
    elif page == "Advisory":
        advisory_page()

    st.markdown(
        '<div class="app-footer">Developed for Data Science Hackathon 2026</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
