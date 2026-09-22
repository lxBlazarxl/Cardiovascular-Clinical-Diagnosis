

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
