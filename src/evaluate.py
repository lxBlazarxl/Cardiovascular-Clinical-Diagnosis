import json

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src import config, data_loader


def _sensitivity(y_true, y_pred):
    _, _, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) else 0.0


def _specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def _net_benefit(y_true, proba, thresholds):
    n = len(y_true)
    prevalence = float(np.mean(y_true))
    benefits = []
    for pt in thresholds:
        pred = (proba >= pt).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        nb = tp / n - fp / n * (pt / (1 - pt))
        benefits.append(nb)
    treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
    return np.array(benefits), np.array(treat_all)


def plot_roc(y, proba, threshold, path):
    fpr, tpr, _ = roc_curve(y, proba)
    pred = (proba >= threshold).astype(int)
    sens = _sensitivity(y, pred)
    spec = _specificity(y, pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc_score(y, proba):.3f})")
    ax.scatter([1 - spec], [sens], color="#d62728", zorder=5,
               label=f"Operating point (sens={sens:.3f}, spec={spec:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pr(y, proba, path):
    precision, recall, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#2ca02c", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.axhline(float(np.mean(y)), color="k", ls="--", lw=1, label=f"Baseline ({np.mean(y):.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_calibration(y, calibrated_proba, uncalibrated_proba, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for proba, label, color in [
        (uncalibrated_proba, f"Uncalibrated (Brier={brier_score_loss(y, uncalibrated_proba):.3f})", "#ff7f0e"),
        (calibrated_proba, f"Calibrated (Brier={brier_score_loss(y, calibrated_proba):.3f})", "#1f77b4"),
    ]:
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, color=color, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency (fraction positive)")
    ax.set_title("Calibration / Reliability Diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_decision_curve(y, proba, path):
    thresholds = np.arange(0.10, 0.501, 0.01)
    model_nb, treat_all_nb = _net_benefit(y, proba, thresholds)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(thresholds, model_nb, color="#1f77b4", lw=2, label="Model")
    ax.plot(thresholds, treat_all_nb, color="#d62728", lw=2, ls="--", label="Treat All")
    ax.axhline(0.0, color="k", lw=1, ls=":", label="Treat None")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis")
    ax.set_ylim(bottom=min(-0.05, float(model_nb.min()) - 0.02))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(config.MODEL_ARTIFACT_PATH)
    calibrated = artifact["calibrated_model"]
    base_model = artifact["base_model"]
    transform_pipeline = artifact["transform_pipeline"]
    threshold = artifact["threshold"]

    df = data_loader.load_raw_data()
    X, y = data_loader.split_features_target(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )
    Xt_test = transform_pipeline.transform(X_test)

    calibrated_proba = calibrated.predict_proba(Xt_test)[:, 1]
    uncalibrated_proba = base_model.predict_proba(Xt_test)[:, 1]

    plot_roc(y_test, calibrated_proba, threshold, config.ARTIFACTS_DIR / "roc_curve.png")
    plot_pr(y_test, calibrated_proba, config.ARTIFACTS_DIR / "pr_curve.png")
    plot_calibration(
        y_test, calibrated_proba, uncalibrated_proba,
        config.ARTIFACTS_DIR / "calibration_curve.png",
    )
    plot_decision_curve(y_test, calibrated_proba, config.ARTIFACTS_DIR / "decision_curve.png")

    pred = (calibrated_proba >= threshold).astype(int)
    report = {
        "model": artifact["best_name"],
        "threshold": float(threshold),
        "n_test": int(len(y_test)),
        "roc_auc": float(roc_auc_score(y_test, calibrated_proba)),
        "pr_auc": float(average_precision_score(y_test, calibrated_proba)),
        "sensitivity": float(_sensitivity(y_test, pred)),
        "specificity": float(_specificity(y_test, pred)),
        "brier_calibrated": float(brier_score_loss(y_test, calibrated_proba)),
        "brier_uncalibrated": float(brier_score_loss(y_test, uncalibrated_proba)),
    }
    config.EVALUATION_METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
