import os
import io
import pickle
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
# UI
# -----------------------------

st.set_page_config(page_title="Real-time Model Inference", layout="wide")
st.title("Real-time Prediction App (Streamlit)")

with st.sidebar:
    st.header("Settings")
    default_model_path = "ensemble_voting_top_hybrid_tuned_best.pkl"
    default_scaler_path = "scaler_hybrid.pkl"   # your saved scaler

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


tabs = st.tabs(["Single Prediction", "Batch Prediction"])

# -----------------------------
# Single Prediction (Paste OR Manual, better UI)
# -----------------------------
with tabs[0]:
    st.subheader("Single Prediction (Paste or Manual Input)")
    if model_obj is None:
        st.info("Load a model from the sidebar to begin.")
    else:
        feature_names = try_extract_input_feature_names(model_obj)
        if not feature_names:
            st.error("Model does not expose feature_names_in_. Please save & load the same pipeline used in training.")
        else:
            # Paste box
            st.markdown("**Paste values (space-separated) to auto-fill the fields below:**")
            st.caption(f"Expected order: `{', '.join(feature_names)}`")
            pasted_row = st.text_area(
                "Paste values here (CTRL+Enter to Submit)", 
                placeholder="59 42.65 1 51.33 45.8 66.75 32.7 ..."
            )

            pasted_vals = None
            if pasted_row.strip():
                vals = [x.strip() for x in pasted_row.split() if x.strip() != ""]
                if len(vals) == len(feature_names):
                    try:
                        pasted_vals = [float(v) for v in vals]
                        st.success("Values parsed and loaded into fields below.")
                    except Exception:
                        st.error("Could not convert pasted values to numbers.")
                else:
                    st.warning(f"Expected {len(feature_names)} values, got {len(vals)}")

            # Manual fields (auto-filled if pasted values available)
            st.markdown("**Or enter values manually:**")
            cols = st.columns(min(4, len(feature_names)))
            user_vals = {}
            for i, feat in enumerate(feature_names):
                with cols[i % len(cols)]:
                    default_val = "" if pasted_vals is None else str(pasted_vals[i])
                    user_input = st.text_input(f"🔹 {feat}", value=default_val)
                    user_vals[feat] = user_input

            # Colorful predict button
            predict_clicked = st.button("Predict", type="primary")

            if predict_clicked:
                try:
                    # Convert all inputs
                    clean_vals = {}
                    for k, v in user_vals.items():
                        if v == "" or v is None:
                            raise ValueError(f"Missing value for feature: {k}")
                        clean_vals[k] = float(v)

                    input_df = pd.DataFrame([clean_vals], columns=feature_names)

                    # Apply scaler
                    if 'scaler' not in globals() or scaler is None:
                        st.warning("Scaler not loaded. Using raw values.")
                        X_df = input_df
                    else:
                        X_scaled = scaler.transform(input_df)
                        X_df = pd.DataFrame(X_scaled, columns=feature_names, index=[0])

                    # Predict
                    out = predict_with_model(model_obj, X_df)

                    # Show prediction
                    pred0 = int(out["pred"][0]) if isinstance(out["pred"][0], (np.integer, int)) else out["pred"][0]
                    st.success(f"**Predicted class: {pred0}**")

                    # Show only prob_0 and prob_1 clearly
                    proba = out.get("proba")
                    if proba is not None and proba.shape[1] == 2:
                        prob_df = pd.DataFrame([proba[0]], columns=["prob_0", "prob_1"])
                        st.markdown("Class probabilities")
                        st.dataframe(prob_df.style.highlight_max(axis=1, color="lightgreen"), use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# -----------------------------
# Batch Prediction (with saved scaler)
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
                    if scaler is None:
                        st.error("Scaler not loaded! Upload scaler_hybrid.pkl")
                    else:
                        # Scale using training scaler
                        X_scaled = scaler.transform(df)
                        df_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)

                        # Predict
                        out = predict_with_model(model_obj, df_scaled)

                        # Build result
                        result = pd.DataFrame()
                        result["prediction"] = out["pred"]

                        if "proba" in out:
                            for i in range(out["proba"].shape[1]):
                                result[f"prob_{i}"] = out["proba"][:, i]

                        st.success("Predictions completed.")
                        st.dataframe(result.head(20), use_container_width=True)
                        df_download_button(result, "predictions.csv", "Download predictions")
                except Exception as e:
                    st.error(f"Prediction failed. Error: {e}")
