PALETTE = {
    "bg": "#0f1420",
    "panel": "#1a2233",
    "panel_alt": "#222c40",
    "border": "#2e3a52",
    "text": "#e8edf7",
    "muted": "#94a3b8",
    "accent": "#4f8cff",
    "danger": "#ef4444",
    "safe": "#22c55e",
    "warn": "#f59e0b",
}

GLOBAL_CSS = """
<style>
  .stApp { background: #0f1420; }
  section[data-testid="stSidebar"] { background: #141b29; border-right: 1px solid #2e3a52; }
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

  .card {
    background: #1a2233; border: 1px solid #2e3a52; border-radius: 14px;
    padding: 20px 22px; margin-bottom: 16px;
  }
  .card h3 { margin: 0 0 4px 0; font-size: 15px; letter-spacing: .04em;
             text-transform: uppercase; color: #94a3b8; font-weight: 600; }
  .card .sub { color: #94a3b8; font-size: 13px; margin: 0; }

  .verdict {
    border-radius: 16px; padding: 26px 30px; margin-bottom: 18px;
    display: flex; align-items: center; justify-content: space-between;
    gap: 24px; border: 1px solid;
  }
  .verdict.high { background: #3b1518; border-color: #7f1d1d; }
  .verdict.low  { background: #12291d; border-color: #14532d; }
  .verdict .left { display: flex; align-items: center; gap: 18px; }
  .verdict .icon { font-size: 40px; line-height: 1; }
  .verdict .label { font-size: 26px; font-weight: 700; margin: 0; }
  .verdict.high .label { color: #fca5a5; }
  .verdict.low  .label { color: #86efac; }
  .verdict .desc { margin: 4px 0 0 0; font-size: 14px; color: #cbd5e1; max-width: 620px; }
  .verdict .prob { text-align: right; }
  .verdict .prob .num { font-size: 44px; font-weight: 800; line-height: 1; }
  .verdict.high .prob .num { color: #f87171; }
  .verdict.low  .prob .num { color: #4ade80; }
  .verdict .prob .cap { font-size: 12px; color: #94a3b8; text-transform: uppercase; }

  .gauge-track {
    position: relative; height: 22px; border-radius: 11px; overflow: hidden;
    background: linear-gradient(90deg,#14532d 0%,#3f6212 38%,#854d0e 58%,#7f1d1d 100%);
    border: 1px solid #2e3a52;
  }
  .gauge-marker {
    position: absolute; top: -5px; width: 3px; height: 32px;
    background: #e8edf7; box-shadow: 0 0 8px rgba(232,237,247,.8);
  }
  .gauge-threshold {
    position: absolute; top: -3px; width: 2px; height: 28px;
    background: repeating-linear-gradient(#f59e0b 0 4px, transparent 4px 8px);
  }
  .gauge-scale { display: flex; justify-content: space-between; margin-top: 6px;
                 font-size: 11px; color: #94a3b8; }

  .metric-row { display: flex; gap: 14px; flex-wrap: wrap; }
  .metric-box {
    flex: 1; min-width: 150px; background: #222c40; border: 1px solid #2e3a52;
    border-radius: 12px; padding: 14px 16px;
  }
  .metric-box .k { font-size: 11px; text-transform: uppercase; color: #94a3b8; }
  .metric-box .v { font-size: 24px; font-weight: 700; margin-top: 4px; }
  .metric-box .h { font-size: 12px; color: #94a3b8; margin-top: 2px; }

  .contrib-row { display: flex; align-items: center; gap: 12px; margin-bottom: 7px; }
  .contrib-name { width: 210px; font-size: 13px; color: #cbd5e1; text-align: right; flex-shrink: 0; }
  .contrib-track { flex: 1; position: relative; height: 20px; }
  .contrib-center { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #2e3a52; }
  .contrib-bar { position: absolute; top: 3px; height: 14px; border-radius: 3px; }
  .contrib-bar.up { background: #ef4444; left: 50%; }
  .contrib-bar.down { background: #3b82f6; right: 50%; }
  .contrib-val { width: 62px; font-size: 12px; color: #94a3b8; flex-shrink: 0; }

  .ba-wrap { display: flex; align-items: flex-end; gap: 40px; justify-content: center;
             padding: 10px 0 4px 0; }
  .ba-col { text-align: center; }
  .ba-bar { width: 90px; border-radius: 8px 8px 0 0; margin: 0 auto; }
  .ba-num { font-size: 22px; font-weight: 700; margin-top: 8px; }
  .ba-cap { font-size: 12px; color: #94a3b8; text-transform: uppercase; }
  .ba-arrow { font-size: 30px; color: #64748b; padding-bottom: 42px; }

  .step { display: flex; gap: 14px; margin-bottom: 14px; }
  .step .n { flex-shrink: 0; width: 26px; height: 26px; border-radius: 50%;
             background: #4f8cff; color: #fff; font-size: 13px; font-weight: 700;
             display: flex; align-items: center; justify-content: center; }
  .step .t { font-size: 14px; color: #cbd5e1; line-height: 1.5; }
  .step .t b { color: #e8edf7; }
</style>
"""

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
    "cp_1.0": "Chest pain: typical angina",
    "cp_2.0": "Chest pain: atypical angina",
    "cp_3.0": "Chest pain: non-anginal",
    "cp_4.0": "Chest pain: asymptomatic",
    "restecg_0.0": "ECG: normal",
    "restecg_1.0": "ECG: ST-T abnormality",
    "restecg_2.0": "ECG: LV hypertrophy",
    "slope_1.0": "ST slope: upsloping",
    "slope_2.0": "ST slope: flat",
    "slope_3.0": "ST slope: downsloping",
    "thal_3.0": "Thallium: normal",
    "thal_6.0": "Thallium: fixed defect",
    "thal_7.0": "Thallium: reversible defect",
}


def pretty_name(raw):
    return PRETTY.get(raw, raw.replace("_", " ").capitalize())


def verdict_html(risk, threshold):
    high = risk >= threshold
    cls = "high" if high else "low"
    icon = "&#9888;&#65039;" if high else "&#9989;"
    label = "HIGH RISK" if high else "LOW / MODERATE RISK"
    desc = (
        "This patient's profile meets the referral threshold. Invasive coronary "
        "angiography should be considered."
        if high
        else "This patient's profile is below the referral threshold. Continue "
        "non-invasive monitoring and risk-factor control."
    )
    return (
        f'<div class="verdict {cls}">'
        f'<div class="left"><div class="icon">{icon}</div>'
        f'<div><p class="label">{label}</p><p class="desc">{desc}</p></div></div>'
        f'<div class="prob"><div class="num">{risk * 100:.1f}%</div>'
        f'<div class="cap">CAD probability</div></div></div>'
    )


def gauge_html(risk, threshold):
    pos = max(0.0, min(1.0, risk)) * 100
    tpos = max(0.0, min(1.0, threshold)) * 100
    return (
        f'<div class="gauge-track">'
        f'<div class="gauge-marker" style="left:calc({pos:.1f}% - 1px);"></div>'
        f'<div class="gauge-threshold" style="left:{tpos:.1f}%;"></div></div>'
        f'<div class="gauge-scale"><span>0% &middot; low</span>'
        f'<span style="color:#f59e0b;">threshold {threshold * 100:.1f}%</span>'
        f'<span>100% &middot; high</span></div>'
    )


def metrics_html(risk, threshold, sensitivity, specificity):
    boxes = [
        ("CAD probability", f"{risk * 100:.1f}%", "calibrated (Platt)"),
        ("Decision threshold", f"{threshold * 100:.1f}%", "tuned for sensitivity"),
        ("Test sensitivity", f"{sensitivity * 100:.1f}%", "true positives caught"),
        ("Test specificity", f"{specificity * 100:.1f}%", "true negatives cleared"),
    ]
    inner = "".join(
        f'<div class="metric-box"><div class="k">{k}</div>'
        f'<div class="v">{v}</div><div class="h">{h}</div></div>'
        for k, v, h in boxes
    )
    return f'<div class="metric-row">{inner}</div>'


def contributions_html(shap_values, feature_names, top_n=10):
    pairs = sorted(
        zip(feature_names, shap_values), key=lambda p: abs(p[1]), reverse=True
    )[:top_n]
    scale = max((abs(v) for _, v in pairs), default=1.0) or 1.0
    rows = []
    for name, value in pairs:
        width = abs(value) / scale * 50.0
        direction = "up" if value > 0 else "down"
        rows.append(
            f'<div class="contrib-row">'
            f'<div class="contrib-name">{pretty_name(name)}</div>'
            f'<div class="contrib-track"><div class="contrib-center"></div>'
            f'<div class="contrib-bar {direction}" style="width:{width:.1f}%;"></div></div>'
            f'<div class="contrib-val">{value:+.3f}</div></div>'
        )
    return "".join(rows)


def summary_html(shap_values, feature_names, top_n=3):
    pairs = sorted(
        zip(feature_names, shap_values), key=lambda p: abs(p[1]), reverse=True
    )[:top_n]
    ups = [pretty_name(n) for n, v in pairs if v > 0]
    downs = [pretty_name(n) for n, v in pairs if v < 0]
    parts = []
    if ups:
        parts.append("<b>Increasing risk:</b> " + ", ".join(ups))
    if downs:
        parts.append("<b>Decreasing risk:</b> " + ", ".join(downs))
    return "<br>".join(parts) if parts else "No dominant contributors."


def before_after_html(original, revised):
    def bar(p, color):
        return f'<div class="ba-bar" style="height:{max(p * 220, 4):.0f}px;background:{color};"></div>'

    def color(p):
        return "#ef4444" if p >= 0.5 else "#f59e0b" if p >= 0.25 else "#22c55e"

    return (
        f'<div class="ba-wrap">'
        f'<div class="ba-col">{bar(original, color(original))}'
        f'<div class="ba-num">{original * 100:.1f}%</div>'
        f'<div class="ba-cap">Current</div></div>'
        f'<div class="ba-arrow">&#8594;</div>'
        f'<div class="ba-col">{bar(revised, color(revised))}'
        f'<div class="ba-num">{revised * 100:.1f}%</div>'
        f'<div class="ba-cap">After intervention</div></div></div>'
    )
