import json
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np

from src import config

a = joblib.load(config.MODEL_ARTIFACT_PATH)
base = a["base_model"]
tp = a["transform_pipeline"]
pre = tp.named_steps["preprocessor"]
cal = a["calibrated_model"]

out = {
    "coef": base.coef_.ravel().tolist(),
    "intercept": float(base.intercept_[0]),
    "classes": base.classes_.tolist(),
    "feature_names": list(a["feature_names"]),
    "threshold": float(a["threshold"]),
    "calibrator_coef": cal.calibrator_.coef_.ravel().tolist(),
    "calibrator_intercept": float(cal.calibrator_.intercept_[0]),
}

num_pipe = pre.named_steps.get("num") if hasattr(pre, "named_steps") else None
for name, tr, cols in pre.transformers_:
    if name == "num":
        out["num_median"] = tr.named_steps["imputer"].statistics_.tolist()
        out["num_mean"] = tr.named_steps["scaler"].mean_.tolist()
        out["num_scale"] = tr.named_steps["scaler"].scale_.tolist()
    if name == "bin":
        out["bin_median"] = tr.named_steps["imputer"].statistics_.tolist()
        out["bin_mean"] = tr.named_steps["scaler"].mean_.tolist()
        out["bin_scale"] = tr.named_steps["scaler"].scale_.tolist()
    if name == "cat":
        out["cat_categories"] = [c.tolist() for c in tr.named_steps["encoder"].categories_]
        out["cat_cols"] = list(cols)

with open("artifacts/model_export.json", "w") as f:
    json.dump(out, f, indent=2)
