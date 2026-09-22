
CP_PRETTY = {
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
PRETTY.update(CP_PRETTY)


def verdict_html(risk, threshold):
    high = risk >= threshold
    cls = "high" if high else "low"
    icon = "&#9888;&#65039;" if high else "&#9989;"
    label = "HIGH RISK" if high else "LOW / MODERATE RISK"
    if high:
        desc = ("This patient meets the referral threshold. "
                "Invasive coronary angiography should be considered.")
    else:
        desc = ("This patient is below the referral threshold. Continue "
                "non-invasive monitoring and risk-factor control.")
    return (
        f'<div class="verdict {cls}">'
        f'<div class="left"><div class="icon">{icon}</div>'
        f'<div><p class="label">{label}</p>'
        f'<p class="desc">{desc}</p></div></div>'
        f'<div class="prob"><div class="num">{risk * 100:.1f}%</div>'
        f'<div class="cap">CAD probability</div></div></div>'
    )
