from robo.eval.audit_metrics import compute, risk_at_coverage, risk_coverage_curve


def test_perfect_classifier_metrics():
    labels = [0, 0, 1, 1]
    scores = [0.0, 0.1, 0.9, 1.0]
    result = compute(labels, scores)
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert risk_at_coverage(labels, scores, 0.5) == 0.0
    assert [row["risk"] for row in risk_coverage_curve(labels, scores)] == [
        0.0, 0.0, 1 / 3, 0.5]
