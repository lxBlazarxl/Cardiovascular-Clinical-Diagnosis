import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["trestbps"] = pd.to_numeric(X["trestbps"], errors="coerce")
        X["thalach"] = pd.to_numeric(X["thalach"], errors="coerce")
        X["age"] = pd.to_numeric(X["age"], errors="coerce")
        X["rpp"] = (X["trestbps"] * X["thalach"]) / config.RPP_SCALE
        X["hr_reserve"] = X["thalach"] / (config.AGE_MAX_HR - X["age"])
        X["hr_reserve"] = X["hr_reserve"].replace([np.inf, -np.inf], np.nan)
        return X


def build_preprocessor() -> ColumnTransformer:
    numeric_features = config.NUMERIC_COLUMNS + config.ENGINEERED_FEATURES
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, config.CATEGORICAL_COLUMNS),
            ("bin", binary_pipeline, config.BINARY_COLUMNS),
        ],
        remainder="drop",
    )


def build_full_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("engineer", ClinicalFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_feature_names(preprocessor) -> list:
    return [
        name.split("__", 1)[1] if "__" in name else name
        for name in preprocessor.get_feature_names_out()
    ]
