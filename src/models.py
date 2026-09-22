from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config


def build_logistic_regression():
    return LogisticRegression(
        solver="saga",
        l1_ratio=0.5,
        max_iter=1000,
        random_state=config.RANDOM_STATE,
    )


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=config.RANDOM_STATE,
    )


def build_xgboost():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        random_state=config.RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=4,
        tree_method="hist",
    )


def candidate_models() -> dict:
    return {
        "logistic_regression": build_logistic_regression,
        "random_forest": build_random_forest,
        "xgboost": build_xgboost,
    }
