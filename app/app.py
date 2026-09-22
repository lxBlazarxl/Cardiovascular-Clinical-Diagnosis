import json
import sys
from pathlib import Path

import joblib
import matplotlib
import streamlit as st

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.explainer import CardiacExplainer  # noqa: E402


@st.cache_resource
def load_artifact():
    artifact = joblib.load(config.MODEL_ARTIFACT_PATH)
    return artifact, CardiacExplainer(artifact)


@st.cache_data
def load_metrics():
    if config.EVALUATION_METRICS_PATH.exists():
        return json.loads(config.EVALUATION_METRICS_PATH.read_text())
    return {}


HIGH_RISK = {
    "age": 58, "sex": 1, "cp": 4, "trestbps": 158, "chol": 285, "fbs": 0,
    "restecg": 2, "thalach": 118, "exang": 1, "oldpeak": 2.6, "slope": 2,
    "ca": 2, "thal": 7,
}
LOW_RISK = {
    "age": 41, "sex": 0, "cp": 2, "trestbps": 118, "chol": 168, "fbs": 0,
    "restecg": 0, "thalach": 176, "exang": 0, "oldpeak": 0.0, "slope": 1,
    "ca": 0, "thal": 3,
}

CP_LABELS = {
    1: "1 - Typical angina",
    2: "2 - Atypical angina",
    3: "3 - Non-anginal pain",
    4: "4 - Asymptomatic",
}
RESTECG_LABELS = {0: "0 - Normal", 1: "1 - ST-T wave abnormality", 2: "2 - LV hypertrophy"}
SLOPE_LABELS = {1: "1 - Upsloping", 2: "2 - Flat", 3: "3 - Downsloping"}
THAL_LABELS = {3: "3 - Normal", 6: "6 - Fixed defect", 7: "7 - Reversible defect"}


def init_state():
    defaults = dict(HIGH_RISK)
    defaults.update({"cf_bp": 120, "cf_chol": 180, "cf_thalach": None})
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_sample(sample):
    for key, value in sample.items():
        st.session_state[key] = value
    st.session_state["cf_bp"] = sample["trestbps"]
    st.session_state["cf_chol"] = sample["chol"]


def collect_patient():
    return {
        "age": st.session_state.age,
        "sex": int(st.session_state.sex),
        "cp": st.session_state.cp,
        "trestbps": st.session_state.trestbps,
        "chol": st.session_state.chol,
        "fbs": int(st.session_state.fbs),
        "restecg": st.session_state.restecg,
        "thalach": st.session_state.thalach,
        "exang": int(st.session_state.exang),
        "oldpeak": st.session_state.oldpeak,
        "slope": st.session_state.slope,
        "ca": st.session_state.ca,
        "thal": st.session_state.thal,
    }


def sidebar_form():
    st.sidebar.title("Patient Input")
    st.sidebar.button("Load Sample High-Risk Patient", on_click=load_sample, args=(HIGH_RISK,), use_container_width=True)
    st.sidebar.button("Load Sample Low-Risk Patient", on_click=load_sample, args=(LOW_RISK,), use_container_width=True)
    st.sidebar.divider()

    st.sidebar.subheader("Demographics")
    st.sidebar.slider("Age (years)", 25, 85, key="age")
    st.sidebar.radio("Biological Sex", [0, 1], key="sex", format_func=lambda v: "Male" if v == 1 else "Female", horizontal=True)

    st.sidebar.subheader("Symptoms & History")
    st.sidebar.selectbox("Chest Pain Type", list(CP_LABELS), key="cp", format_func=lambda v: CP_LABELS[v])
    st.sidebar.checkbox("Fasting Blood Sugar > 120 mg/dl", key="fbs")

    st.sidebar.subheader("Vitals & Labs")
    st.sidebar.slider("Resting Blood Pressure (mmHg)", 90, 200, key="trestbps")
    st.sidebar.slider("Serum Cholesterol (mg/dl)", 120, 500, key="chol")

    st.sidebar.subheader("Diagnostic Tests")
    st.sidebar.selectbox("Resting ECG", list(RESTECG_LABELS), key="restecg", format_func=lambda v: RESTECG_LABELS[v])
    st.sidebar.slider("Max Heart Rate Achieved (bpm)", 70, 210, key="thalach")
    st.sidebar.checkbox("Exercise Induced Angina", key="exang")
    st.sidebar.slider("ST Depression (oldpeak, mm)", 0.0, 6.0, key="oldpeak", step=0.1)
    st.sidebar.selectbox("Peak Exercise ST Slope", list(SLOPE_LABELS), key="slope", format_func=lambda v: SLOPE_LABELS[v])
    st.sidebar.slider("Fluoroscopy Vessels (ca)", 0, 3, key="ca")
    st.sidebar.selectbox("Thallium Scan", list(THAL_LABELS), key="thal", format_func=lambda v: THAL_LABELS[v])


def main():
    st.set_page_config(page_title="Coronary Artery Disease Triage", layout="wide")
    init_state()

    artifact, explainer = load_artifact()
    metrics = load_metrics()
    threshold = float(artifact["threshold"])

    st.title("Coronary Artery Disease Clinical Triage & Explainable AI")
    badge_cols = st.columns(4)
    badge_cols[0].metric("Model", artifact["best_name"].replace("_", " ").title())
    badge_cols[1].metric("Operating Threshold", f"{threshold:.3f}")
    badge_cols[2].metric("Test Sensitivity", f"{metrics.get('sensitivity', 0) * 100:.1f}%")
    badge_cols[3].metric("Calibration", "Platt (sigmoid)")

    sidebar_form()
    patient = collect_patient()

    risk = float(explainer.calibrated_model.predict_proba(explainer._transform(patient))[0, 1])
    high_risk = risk >= threshold

    st.divider()
    top = st.columns(3)
    top[0].metric("Calibrated CAD Risk", f"{risk * 100:.2f}%")
    top[1].metric("Triage Status", "HIGH RISK" if high_risk else "LOW / MODERATE")
    top[2].metric("Operating Threshold", f"{threshold:.3f}")
    st.caption(
        f"Risk updates live as sliders move. At very high risk the calibrated curve saturates, "
        f"so large input changes may move the display by only fractions of a percent."
    )

    if high_risk:
        st.error("HIGH RISK - Urgent catheterization referral recommended.")
    else:
        st.success("LOW / MODERATE RISK - Continue non-invasive monitoring and risk-factor control.")

    tab_explain, tab_whatif, tab_bench = st.tabs(
        ["Point-of-Care Explainability", "Actionable What-If Intervention Simulator", "Model Benchmarking & Calibration"]
    )

    with tab_explain:
        st.subheader("Individual SHAP Waterfall")
        fig, _ = explainer.waterfall(patient)
        st.pyplot(fig, use_container_width=True)

    with tab_whatif:
        st.subheader("Simulate Modifiable Risk Factors")
        st.slider(
            "Target Resting Blood Pressure (mmHg)",
            90,
            200,
            key="cf_bp",
        )
        st.slider(
            "Target Serum Cholesterol (mg/dl)",
            120,
            500,
            key="cf_chol",
        )
        cf = explainer.simulate_counterfactual(
            patient,
            target_bp=st.session_state["cf_bp"],
            target_chol=st.session_state["cf_chol"],
        )
        cols = st.columns(3)
        cols[0].metric("Current Risk", f"{cf['original_risk'] * 100:.2f}%")
        cols[1].metric(
            "Revised Risk",
            f"{cf['revised_risk'] * 100:.2f}%",
            delta=f"{(cf['revised_risk'] - cf['original_risk']) * 100:+.2f} pp",
        )
        cols[2].metric("Absolute Reduction", f"{cf['absolute_reduction'] * 100:.2f} pp")
        st.progress(min(max(cf["revised_risk"], 0.0), 1.0), text=f"Revised risk {cf['revised_risk'] * 100:.2f}%")
        if cf["absolute_reduction"] > 0:
            st.info(f"Relative risk reduction of {cf['relative_reduction_pct']:.2f}% from the simulated intervention.")
        else:
            st.warning("The simulated targets do not reduce this patient's modeled risk.")

    with tab_bench:
        st.subheader("Evaluation Curves")
        cols = st.columns(2)
        for col, (name, caption) in zip(
            cols * 2,
            [
                ("roc_curve.png", "ROC Curve"),
                ("pr_curve.png", "Precision-Recall Curve"),
                ("calibration_curve.png", "Calibration / Reliability Diagram"),
                ("decision_curve.png", "Decision Curve Analysis"),
            ],
        ):
            path = config.ARTIFACTS_DIR / name
            if path.exists():
                col.image(str(path), caption=caption, use_container_width=True)
        if metrics:
            st.json(metrics)


if __name__ == "__main__":
    main()
