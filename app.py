import streamlit as st
import numpy as np
import pandas as pd
import pickle, json, pathlib

# ---------- Page setup ----------
st.set_page_config(page_title="Parkinson's Prediction", page_icon="🧠", layout="centered")
st.title("🧠 Parkinson's Disease Prediction (UCI Voice Dataset)")
st.caption("Learning/demo only — not medical advice.")

# ---------- Sidebar ----------
st.sidebar.title("About this app")
st.sidebar.info(
    "Interactive demo using a scikit-learn model trained on the UCI Parkinson’s voice dataset. "
    "Provide feature values to estimate the likelihood of Parkinson’s disease."
)

# Optional threshold to convert probability -> label (only used if your model has predict_proba)
threshold = st.sidebar.slider("Decision threshold (probability)", 0.0, 1.0, 0.50, 0.01)

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    with open("artifacts/parkinsons_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("artifacts/feature_names.json", "r") as f:
        feature_names = json.load(f)
    with open("artifacts/feature_stats.json", "r") as f:
        stats = json.load(f)
    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, feature_names, stats, scaler

try:
    model, feature_names, stats, scaler = load_artifacts()
except Exception as e:
    st.error(f"Artifact load failed: {e}")
    st.stop()

# ---------- Helpers ----------
def load_sample(values_dict):
    """Preload UI inputs from a dict {feature: value} using session_state keys."""
    for i, feat in enumerate(feature_names):
        key = f"inp_{i}"
        if feat in values_dict:
            st.session_state[key] = float(values_dict[feat])

def means_dict():
    return {k: float(stats[k]["mean"]) for k in feature_names if k in stats}

# A reasonable sample row (feel free to swap with a real example from your dataset)
SAMPLE_ROW = {
    "MDVP:Fo(Hz)": 154.229, "MDVP:Fhi(Hz)": 197.105, "MDVP:Flo(Hz)": 116.325,
    "MDVP:Jitter(%)": 0.00622, "MDVP:Jitter(Abs)": 0.000044, "MDVP:RAP": 0.003306,
    "MDVP:PPQ": 0.003446, "Jitter:DDP": 0.009920, "MDVP:Shimmer": 0.029709,
    "MDVP:Shimmer(dB)": 0.282251, "Shimmer:APQ3": 0.015664, "Shimmer:APQ5": 0.017878,
    "MDVP:APQ": 0.024081, "Shimmer:DDA": 0.046993, "NHR": 0.024847, "HNR": 21.886,
    "RPDE": 0.498536, "DFA": 0.718099, "spread1": -5.684397, "spread2": 0.226510,
    "D2": 2.381826, "PPE": 0.206552
}

# ---------- Feature list ----------
with st.expander("Expected features (in order)"):
    st.code("\n".join(feature_names))

# ---------- Top controls ----------
top_cols = st.columns([1, 1, 2])
with top_cols[0]:
    if st.button("Load sample input"):
        load_sample(SAMPLE_ROW)
with top_cols[1]:
    if st.button("Reset to means"):
        load_sample(means_dict())

# ---------- Inputs ----------
st.subheader("Enter feature values")
cols = st.columns(3)
values = []
for i, feat in enumerate(feature_names):
    s = stats.get(feat, {"min": 0.0, "max": 1e6, "mean": 0.0})
    with cols[i % 3]:
        val = st.number_input(
            feat,
            min_value=float(s["min"]),
            max_value=float(s["max"]),
            value=float(s["mean"]),
            format="%.6f",
            key=f"inp_{i}"  # important for sample/reset
        )
        values.append(val)

# ---------- Predict ----------
if st.button("Predict"):
    try:
        X_df = pd.DataFrame([values], columns=feature_names)

        # Apply the same preprocessing used during training
        X_scaled = scaler.transform(X_df)

        # Probability (if available) and prediction
        if hasattr(model, "predict_proba"):
            proba_pos = float(model.predict_proba(X_scaled)[:, 1][0])
            pred = int(proba_pos >= threshold)
        else:
            proba_pos = None
            pred = int(model.predict(X_scaled)[0])

        # ---------- Result card ----------
        st.subheader("Result")
        if pred == 1:
            st.error("🧬 Parkinson’s likely.")
        else:
            st.success("🌿 Parkinson’s unlikely.")

        if proba_pos is not None:
            st.metric("Estimated Probability", f"{proba_pos:.2%}")
            st.caption(f"Decision threshold: {threshold:.2f}  →  Predicted label: {pred}")

        # ---------- Feature importance (if available) ----------
        if hasattr(model, "feature_importances_"):
            import numpy as np
            import matplotlib.pyplot as plt

            importances = np.array(model.feature_importances_)
            order = np.argsort(importances)[-10:]  # Top 10
            top_feats = [feature_names[i] for i in order]
            top_imps = importances[order]

            st.subheader("Top feature importances")
            fig, ax = plt.subplots()
            ax.barh(range(len(top_feats)), top_imps)
            ax.set_yticks(range(len(top_feats)))
            ax.set_yticklabels(top_feats)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
