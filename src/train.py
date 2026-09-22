import json

import joblib
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src import calibration, config, data_loader, models, preprocessor


def _specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def _sensitivity(y_true, y_pred):
    _, _, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) else 0.0


def cross_validate_candidates(X, y):
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    results = {}
    oof = {}
    for name, factory in models.candidate_models().items():
        pipe = preprocessor.build_full_pipeline(factory())
        proba = cross_val_predict(
            pipe, X, y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
            "sensitivity": float(_sensitivity(y, pred)),
            "specificity": float(_specificity(y, pred)),
            "brier": float(brier_score_loss(y, proba)),
        }
        oof[name] = proba
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    return results, oof, best_name


def find_threshold(y_true, proba, floor=None):
    floor = config.CLINICAL_SENSITIVITY_FLOOR if floor is None else floor
    best = None
    for t in config.THRESHOLD_SEARCH_GRID:
        pred = (proba >= t).astype(int)
        sens = _sensitivity(y_true, pred)
        if sens < floor:
            continue
        spec = _specificity(y_true, pred)
        if best is None or spec > best["specificity"]:
            best = {
                "threshold": float(t),
                "sensitivity": float(sens),
                "specificity": float(spec),
            }
    if best is None:
        fallback = config.DEFAULT_DECISION_THRESHOLD
        pred = (proba >= fallback).astype(int)
        best = {
            "threshold": float(fallback),
            "sensitivity": float(_sensitivity(y_true, pred)),
            "specificity": float(_specificity(y_true, pred)),
        }
    return best


def train_best_model(X, y, best_name, oof_proba):
    transform_pipeline = Pipeline(
        steps=[
            ("engineer", preprocessor.ClinicalFeatureEngineer()),
            ("preprocessor", preprocessor.build_preprocessor()),
        ]
    )
    transform_pipeline.fit(X, y)
    Xt = transform_pipeline.transform(X)

    base_model = models.candidate_models()[best_name]()
    base_model.fit(Xt, y)

    calibrated = calibration.PlattCalibrator(
        base_model, cv=config.CV_FOLDS, random_state=config.RANDOM_STATE
    )
    calibrated.fit(Xt, y)

    proba = calibrated.predict_proba(Xt)[:, 1]
    threshold_info = find_threshold(y, proba)

    uncalibrated_proba = base_model.predict_proba(Xt)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "brier_uncalibrated": float(brier_score_loss(y, uncalibrated_proba)),
    }
    artifact = {
        "transform_pipeline": transform_pipeline,
        "base_model": base_model,
        "calibrated_model": calibrated,
        "best_name": best_name,
        "threshold": threshold_info["threshold"],
        "threshold_metrics": threshold_info,
        "metrics": metrics,
        "feature_names": preprocessor.get_feature_names(
            transform_pipeline.named_steps["preprocessor"]
        ),
    }
    return artifact


def main():
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = data_loader.load_raw_data()
    X, y = data_loader.split_features_target(df)

    results, oof, best_name = cross_validate_candidates(X, y)

    print("Stratified 5-Fold Cross-Validation (out-of-fold)")
    print(
        f"{'model':<22}{'roc_auc':>10}{'pr_auc':>10}{'sens':>8}{'spec':>8}{'brier':>8}"
    )
    for name, m in results.items():
        print(
            f"{name:<22}{m['roc_auc']:>10.4f}{m['pr_auc']:>10.4f}"
            f"{m['sensitivity']:>8.3f}{m['specificity']:>8.3f}{m['brier']:>8.4f}"
        )
    print(f"\nBest model by ROC-AUC: {best_name}")

    artifact = train_best_model(X, y, best_name, oof[best_name])
    joblib.dump(artifact, config.MODEL_ARTIFACT_PATH)
    print(
        f"Saved artifact to {config.MODEL_ARTIFACT_PATH} "
        f"(threshold={artifact['threshold']:.3f}, "
        f"sens={artifact['threshold_metrics']['sensitivity']:.3f}, "
        f"spec={artifact['threshold_metrics']['specificity']:.3f})"
    )

    summary_path = config.ARTIFACTS_DIR / "cv_results.json"
    summary_path.write_text(
        json.dumps(
            {
                "cv": results,
                "best": best_name,
                "artifact": {
                    k: v
                    for k, v in artifact.items()
                    if k not in ("calibrated_model", "base_model", "transform_pipeline")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
