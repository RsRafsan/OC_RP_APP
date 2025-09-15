import os 
import io 
import pickle 
import random
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler    
import numpy as np 
import pandas as pd 
import streamlit as st 
from sklearn.metrics import classification_report, confusion_matrix   

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def try_extract_input_feature_names(model) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for s in model.named_steps.values():
            if hasattr(s, "feature_names_in_"):
                return list(s.feature_names_in_)
    return None

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == object:
            try:
                df2[c] = pd.to_numeric(df2[c])
            except Exception:
                pass
    return df2

def predict_with_model(model, X: pd.DataFrame) -> Dict[str, Any]:
    X = coerce_types(X)
    out: Dict[str, Any] = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        out["proba"] = proba
        out["pred"] = np.argmax(proba, axis=1)
    else:
        out["pred"] = model.predict(X)
    if hasattr(model, "classes_"):
        out["classes_"] = model.classes_
    return out

def df_download_button(df: pd.DataFrame, filename: str, label: str):
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(label, data=csv_buf.getvalue(), file_name=filename, mime="text/csv")

# -----------------------------
# Risk assignment based on High Risk probability
# -----------------------------
def risk_label_from_proba(p_high: float) -> str:
    """Assign risk label purely from high-risk probability"""
    if p_high < 0.40:
        return "Low Risk"
    elif p_high < 0.70:
        return "Moderate Risk"
    else:
        return "High Risk"

def style_risk(val: str) -> str:
    if val == "Low Risk":
        return "background-color:#c8e6c9; color:#1b5e20; font-weight:bold; text-align:center; border-radius:5px;"
    elif val == "Moderate Risk":
        return "background-color:#fff9c4; color:#f57f17; font-weight:bold; text-align:center; border-radius:5px;"
    elif val == "High Risk":
        return "background-color:#ffcdd2; color:#b71c1c; font-weight:bold; text-align:center; border-radius:5px;"
    return ""

# -----------------------------
# SHAP explanation using file
# -----------------------------
def explain_with_shap(input_df, shap_values_df, feature_names, abs_threshold: float = 0.02, top_k: int = 8):
    """Return only HIGH RISK indicators from precomputed SHAP values"""
    if shap_values_df is None:
        return []

    cols = [c for c in feature_names if c in shap_values_df.columns]
    if not cols:
        return []

    mean_abs = shap_values_df[cols].abs().mean()
    mean_signed = shap_values_df[cols].mean()

    # keep only strong & positive (pushing towards high risk)
    strong = mean_abs[mean_abs >= abs_threshold].sort_values(ascending=False)
    if strong.empty:
        strong = mean_abs.sort_values(ascending=False).head(min(top_k, len(cols)))

    explanations = []
    for feat in strong.index:
        if mean_signed[feat] > 0:
            val = float(input_df.iloc[0].get(feat, np.nan))
            explanations.append(f"<p style='font-size:16px;'><b>{feat} = {val:.2f} → High Risk</b></p>")

    return explanations

# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="OvaPredict AI", layout="wide")
st.title("OvaPredict AI: Ovarian Cancer Prediction")

with st.sidebar:
    st.header("Settings")
    default_model_path = "overall_best_federated_xgb.pkl"
    default_scaler_path = "scaler_hybrid.pkl"

    uploaded_model = st.file_uploader("Upload a .pkl model", type=["pkl"])
    uploaded_scaler = st.file_uploader("Upload a .pkl scaler", type=["pkl"])

    # Model
    if uploaded_model is not None:
        with open("uploaded_model.pkl", "wb") as f:
            f.write(uploaded_model.read())
        model_path = "uploaded_model.pkl"
    else:
        model_path = default_model_path

    try:
        model_obj = load_model(model_path)
        if isinstance(model_obj, list):
            st.warning("Model file contained a list. Using the first element.")
            model_obj = model_obj[0]
        st.success(f"Loaded model: {os.path.basename(model_path)}")
    except Exception as e:
        model_obj = None
        st.error(f"Could not load model: {e}")

    # Scaler
    if uploaded_scaler is not None:
        with open("uploaded_scaler.pkl", "wb") as f:
            f.write(uploaded_scaler.read())
        with open("uploaded_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    else:
        try:
            with open(default_scaler_path, "rb") as f:
                scaler = pickle.load(f)
            st.success("Scaler loaded")
        except Exception:
            scaler = None
            st.warning("No scaler found. Upload scaler_hybrid.pkl")

# -----------------------------
# Load dataset medians & SHAP
# -----------------------------
try:
    df = pd.read_csv("selected_features_data.csv")
    medians = df.median(numeric_only=True)
except Exception:
    medians = {}

try:
    shap_values_df = pd.read_csv("shap_values.csv")
    st.sidebar.success("Loaded SHAP values from shap_values.csv")
except Exception:
    shap_values_df = None
    st.sidebar.warning("No shap_values.csv found. Risk indicators will be limited.")

FALLBACK_FEATURE_NAMES = [
    'Age', 'HE4', 'Menopause', 'CA125', 'ALB', 'NEU', 'LYM%', 'ALP',
    'PLT', 'LYM#', 'AST', 'PCT', 'IBIL', 'TBIL', 'CA72-4', 'GLO',
    'MONO#', 'HGB', 'Na', 'CEA', 'Ca', 'GLU.', 'DBIL', 'TP', 'MCH'
]

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["Single Prediction", "Batch Prediction"])

# -----------------------------
# Single Prediction
# -----------------------------
with tabs[0]:
    st.subheader("Single Prediction")

    if model_obj is None:
        st.info("Load a model from the sidebar to begin.")
    else:
        feature_names = try_extract_input_feature_names(model_obj) or FALLBACK_FEATURE_NAMES
        st.markdown("Enter the feature values:")

        if "static_defaults" not in st.session_state:
            st.session_state.static_defaults = {
                feat: float(medians.get(feat, random.uniform(1, 100)))
                for feat in feature_names
            }

        cols = st.columns(min(4, len(feature_names)))
        user_vals = {}

        for i, feat in enumerate(feature_names):
            with cols[i % len(cols)]:
                default_val = st.session_state.static_defaults[feat]
                if feat.lower() == "age":
                    user_input = st.number_input(
                        f"🔹 {feat}",
                        value=int(default_val),
                        step=1,
                        format="%d",
                        key=f"input_{feat}_{i}"
                    )
                else:
                    user_input = st.number_input(
                        f"🔹 {feat}",
                        value=float(default_val),
                        step=0.1,
                        format="%.3f",
                        key=f"input_{feat}_{i}"
                    )
                user_vals[feat] = float(user_input)

        predict_clicked = st.button("Predict", type="primary")

        if predict_clicked:
            try:
                clean_vals = {k: float(v) for k, v in user_vals.items()}
                input_df = pd.DataFrame([clean_vals], columns=feature_names)

                if scaler is None:
                    st.warning("Scaler not loaded. Using raw values.")
                    X_df = input_df
                else:
                    X_scaled = scaler.transform(input_df)
                    X_df = pd.DataFrame(X_scaled, columns=feature_names, index=[0])

                # ✅ always compute probability of High Risk
                proba = model_obj.predict_proba(X_df)[0]
                classes = getattr(model_obj, "classes_", np.array([0, 1]))
                idx_high = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
                p_high = float(proba[idx_high])

                risk_label = risk_label_from_proba(p_high)
                percent = p_high * 100

                st.success("Prediction completed.")
                st.markdown(f"### Prediction Result\n**{risk_label} ({percent:.2f}%)**")

                # ✅ Risk Indicators
                st.markdown("### Risk Indicators (from SHAP)")
                if risk_label == "Low Risk":
                    st.markdown("✅ All features are within normal range.")
                else:
                    explanations = explain_with_shap(input_df, shap_values_df, feature_names)
                    if explanations:
                        st.markdown("".join(explanations), unsafe_allow_html=True)
                    else:
                        st.markdown("✅ No strong high-risk features found.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
# -----------------------------
# Batch Prediction (updated with % formatting)
# -----------------------------
with tabs[1]:
    st.subheader("Batch Prediction")
    if model_obj:
        batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch")
        if batch_file:
            df = pd.read_csv(batch_file)
            st.write("Preview:", df.head())

            if st.button("Predict (Batch)"):
                try:
                    feature_names = try_extract_input_feature_names(model_obj) or FALLBACK_FEATURE_NAMES

                    if scaler is None:
                        st.error("Scaler not loaded! Upload scaler_hybrid.pkl")
                    else:
                        X_scaled = scaler.transform(df[feature_names])
                        df_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=df.index)

                        # Predict probabilities for all rows
                        proba = model_obj.predict_proba(df_scaled)
                        classes = getattr(model_obj, "classes_", np.array([0, 1]))
                        idx_high = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
                        p_high = proba[:, idx_high]

                        # Build results DataFrame
                        result = pd.DataFrame(index=df.index)
                        result["High Risk (%)"] = (p_high * 100).round(2).astype(str) + "%"
                        result["Risk Level"] = [risk_label_from_proba(p) for p in p_high]

                        st.success("Predictions completed.")
                        st.dataframe(
                            result.head(20).style.applymap(style_risk, subset=["Risk Level"]),
                            use_container_width=True
                        )

                        # Download
                        df_download_button(result, "predictions.csv", "Download predictions")

                except Exception as e:
                    st.error(f"Prediction failed. Error: {e}")
