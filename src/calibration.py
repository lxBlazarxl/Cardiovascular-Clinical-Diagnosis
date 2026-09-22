import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src import config


class PlattCalibrator(BaseEstimator, ClassifierMixin):
    """Cross-validated Platt scaling on a model's decision function.

    Equivalent in intent to CalibratedClassifierCV(method='sigmoid') but
    implemented explicitly so it is stable across scikit-learn versions and
    provably sensitive to its input.
    """

    def __init__(self, estimator, cv=None, random_state=None):
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_splits = self.cv if self.cv is not None else config.CV_FOLDS
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )

        oof_scores = np.zeros(len(y), dtype=float)
        for train_idx, valid_idx in skf.split(X, y):
            fold_model = clone(self.estimator)
            fold_model.fit(X[train_idx], y[train_idx])
            oof_scores[valid_idx] = self._score(fold_model, X[valid_idx])

        self.calibrator_ = LogisticRegression(
            C=1e6, solver="lbfgs", max_iter=1000
        )
        self.calibrator_.fit(oof_scores.reshape(-1, 1), y)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self._fitted = True
        return self

    @staticmethod
    def _score(model, X):
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(X)).ravel()
        proba = model.predict_proba(X)[:, 1]
        eps = 1e-9
        proba = np.clip(proba, eps, 1 - eps)
        return np.log(proba / (1 - proba))

    def predict_proba(self, X):
        X = np.asarray(X)
        scores = self._score(self.estimator_, X).reshape(-1, 1)
        return self.calibrator_.predict_proba(scores)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
