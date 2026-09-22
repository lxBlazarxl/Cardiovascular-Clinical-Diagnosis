

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
