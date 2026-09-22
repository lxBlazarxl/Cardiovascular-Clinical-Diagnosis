import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, data_loader, preprocessor
from src.explainer import CardiacExplainer

LOW_RISK_PATIENT = {
    "age": 41, "sex": 0, "cp": 2, "trestbps": 118, "chol": 168, "fbs": 0,
    "restecg": 0, "thalach": 176, "exang": 0, "oldpeak": 0.0, "slope": 1,
    "ca": 0, "thal": 3,
}

HIGH_RISK_PATIENT = {
    "age": 58, "sex": 1, "cp": 4, "trestbps": 158, "chol": 285, "fbs": 0,
    "restecg": 2, "thalach": 118, "exang": 1, "oldpeak": 2.6, "slope": 2,
    "ca": 2, "thal": 7,
}


@pytest.fixture(scope="session")
def artifact():
    assert config.MODEL_ARTIFACT_PATH.exists(), "run `python -m src.train` first"
    return joblib.load(config.MODEL_ARTIFACT_PATH)


@pytest.fixture(scope="session")
def explainer(artifact):
    return CardiacExplainer(artifact)


def test_data_loader():
    df = data_loader.load_raw_data()
    assert df.shape[1] == 14
    assert set(df[config.TARGET_COLUMN].unique()) <= {0, 1}
    assert int(df["ca"].isna().sum()) == 4
    assert int(df["thal"].isna().sum()) == 2
    assert df[config.TARGET_COLUMN].sum() == 139


def test_preprocessor_output_shape():
    df = data_loader.load_raw_data()
    X, _ = data_loader.split_features_target(df)
    pre = preprocessor.build_preprocessor()
    Xe = preprocessor.ClinicalFeatureEngineer().fit_transform(X)
    Xt = pre.fit_transform(Xe)
    names = preprocessor.get_feature_names(pre)
    assert Xt.shape[1] == len(names)
    assert len(names) == len(set(names))
    assert np.isfinite(Xt).all()
    eng = preprocessor.ClinicalFeatureEngineer().fit_transform(X)
    assert np.allclose(eng["rpp"], (X["trestbps"] * X["thalach"]) / 100)
    assert np.allclose(eng["hr_reserve"], X["thalach"] / (220 - X["age"]))


def test_calibrated_probabilities(explainer):
    for patient in (LOW_RISK_PATIENT, HIGH_RISK_PATIENT):
        proba = explainer.calibrated_model.predict_proba(
            explainer._transform(patient)
        )[0, 1]
        assert 0.0 <= proba <= 1.0
        assert np.isfinite(proba)
    low = explainer.calibrated_model.predict_proba(
        explainer._transform(LOW_RISK_PATIENT)
    )[0, 1]
    high = explainer.calibrated_model.predict_proba(
        explainer._transform(HIGH_RISK_PATIENT)
    )[0, 1]
    assert high > low


def test_shap_additivity(explainer):
    Xt = explainer._transform(HIGH_RISK_PATIENT)
    explanation = explainer.explainer(Xt)
    values = np.asarray(explanation.values).ravel()
    base = float(np.ravel(explanation.base_values)[0])
    margin = float(explainer.base_model.decision_function(Xt)[0])
    assert abs(base + values.sum() - margin) < 1e-3


def test_threshold_meets_sensitivity_floor(artifact):
    metrics = artifact["threshold_metrics"]
    assert metrics["sensitivity"] >= config.CLINICAL_SENSITIVITY_FLOOR
    assert artifact["threshold"] < 0.5
