from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_DATA_PATH = DATA_RAW_DIR / "cleveland.csv"
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "best_model.joblib"
EVALUATION_METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

TARGET_COLUMN = "target"

CATEGORICAL_COLUMNS = ["cp", "restecg", "slope", "thal"]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_COLUMNS = ["sex", "fbs", "exang"]

ENGINEERED_FEATURES = ["rpp", "hr_reserve"]

MISSING_TOKEN = "?"

CLINICAL_SENSITIVITY_FLOOR = 0.90
DEFAULT_DECISION_THRESHOLD = 0.35
THRESHOLD_SEARCH_GRID = [round(x, 4) for x in [i / 1000 for i in range(50, 1001)]]

RANDOM_STATE = 42
CV_FOLDS = 5

AGE_MAX_HR = 220
RPP_SCALE = 100
