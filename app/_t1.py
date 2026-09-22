PALETTE = {
    "bg": "#0f1420",
    "panel": "#1a2233",
    "border": "#2e3a52",
    "text": "#e8edf7",
    "muted": "#94a3b8",
    "accent": "#4f8cff",
    "danger": "#ef4444",
    "safe": "#22c55e",
    "warn": "#f59e0b",
}

PRETTY = {
    "age": "Age",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "thalach": "Max heart rate",
    "oldpeak": "ST depression",
    "ca": "Blocked vessels (fluoroscopy)",
    "rpp": "Rate-pressure product",
    "hr_reserve": "Heart-rate reserve",
    "sex": "Male sex",
    "fbs": "High fasting blood sugar",
    "exang": "Exercise-induced angina",
}


def pretty_name(raw):
    return PRETTY.get(raw, raw.replace("_", " ").capitalize())
