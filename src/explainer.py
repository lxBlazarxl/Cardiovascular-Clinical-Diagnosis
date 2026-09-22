import numpy as np
import pandas as pd
import shap

from src import config


class CardiacExplainer:
    def __init__(self, artifact):
        self.artifact = artifact
        self.transform_pipeline = artifact["transform_pipeline"]
        self.base_model = artifact["base_model"]
        self.calibrated_model = artifact["calibrated_model"]
        self.feature_names = artifact.get("feature_names") or self._derive_names()
        self.explainer = self._build_explainer()

    def _derive_names(self):
        return [
            name.split("__", 1)[1] if "__" in name else name
            for name in self.transform_pipeline.named_steps["preprocessor"].get_feature_names_out()
        ]

    def _build_explainer(self):
        if isinstance(self.base_model, tuple()):
            return shap.TreeExplainer(self.base_model)
        linear = ()
        try:
            from sklearn.linear_model import LogisticRegression

            linear = (LogisticRegression,)
        except Exception:
            pass
        if linear and isinstance(self.base_model, linear):
            return shap.LinearExplainer(self.base_model, self._background())
        try:
            from xgboost import XGBClassifier

            if isinstance(self.base_model, XGBClassifier):
                return shap.TreeExplainer(self.base_model)
        except Exception:
            pass
        try:
            from sklearn.ensemble import RandomForestClassifier

            if isinstance(self.base_model, RandomForestClassifier):
                return shap.TreeExplainer(self.base_model)
        except Exception:
            pass
        return shap.LinearExplainer(self.base_model, self._background())

    def _background(self, n=100):
        from src import data_loader

        df = data_loader.load_raw_data()
        X, _ = data_loader.split_features_target(df)
        return self._transform(X.head(n))

    def _transform(self, patient_df):
        if isinstance(patient_df, dict):
            patient_df = pd.DataFrame([patient_df])
        return self.transform_pipeline.transform(patient_df)

    def explain_patient(self, patient_df):
        Xt = self._transform(patient_df)
        explanation = self.explainer(Xt)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[0, :, 1] if values.shape[2] > 1 else values[0, :, 0]
        else:
            values = values[0]
        base_value = explanation.base_values
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.ravel(base_value)[0])
        return {
            "base_value": float(base_value),
            "shap_values": values,
            "feature_names": self.feature_names,
            "features": Xt[0],
            "expected_value": float(base_value),
        }

    def waterfall(self, patient_df):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        info = self.explain_patient(patient_df)
        expl = shap.Explanation(
            values=info["shap_values"],
            base_values=info["base_value"],
            data=info["features"],
            feature_names=info["feature_names"],
        )
        fig = plt.figure()
        shap.plots.waterfall(expl, show=False)
        return fig, info

    def simulate_counterfactual(self, patient_df, target_bp=None, target_chol=None, target_thalach=None):
        if isinstance(patient_df, dict):
            original_df = pd.DataFrame([patient_df])
        else:
            original_df = patient_df.copy()

        original_risk = float(
            self.calibrated_model.predict_proba(self._transform(original_df))[0, 1]
        )

        modified = original_df.copy()
        if target_bp is not None:
            modified["trestbps"] = target_bp
        if target_chol is not None:
            modified["chol"] = target_chol
        if target_thalach is not None:
            modified["thalach"] = target_thalach

        revised_risk = float(
            self.calibrated_model.predict_proba(self._transform(modified))[0, 1]
        )
        return {
            "original_risk": original_risk,
            "revised_risk": revised_risk,
            "absolute_reduction": original_risk - revised_risk,
            "relative_reduction_pct": (
                (original_risk - revised_risk) / original_risk * 100
                if original_risk
                else 0.0
            ),
        }
