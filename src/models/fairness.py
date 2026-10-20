from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "reports" / "test_predictions.csv"
OUTPUT = ROOT / "reports" / "fairness_metrics.csv"


def manual_audit(df, sensitive):
    rows = []
    tprs, fprs, selection_rates = [], [], []
    for group, part in df.groupby(sensitive, dropna=False):
        y, p = part.y_true, part.y_pred
        tp = int(((y == 1) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        selection = float((p == 1).mean())
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        selection_rates.append(selection)
        tprs.append(tpr)
        fprs.append(fpr)
        rows.append({
            sensitive: group,
            "accuracy": accuracy_score(y, p),
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0),
            "selection_rate": selection,
        })
    dp = max(selection_rates) - min(selection_rates)
    eo = max(max(tprs) - min(tprs), max(fprs) - min(fprs))
    out = pd.DataFrame(rows)
    out.insert(0, "protected_attribute", sensitive)
    out["demographic_parity_difference_overall"] = dp
    out["equalized_odds_difference_overall"] = eo
    return out


def audit(df, sensitive):
    try:
        from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate
    except ImportError:
        return manual_audit(df, sensitive)

    frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "precision": lambda y, p: precision_score(y, p, zero_division=0),
            "recall": lambda y, p: recall_score(y, p, zero_division=0),
            "f1": lambda y, p: f1_score(y, p, zero_division=0),
            "selection_rate": selection_rate,
        },
        y_true=df.y_true,
        y_pred=df.y_pred,
        sensitive_features=df[sensitive],
    )
    out = frame.by_group.reset_index()
    out.insert(0, "protected_attribute", sensitive)
    out["demographic_parity_difference_overall"] = demographic_parity_difference(df.y_true, df.y_pred, sensitive_features=df[sensitive])
    out["equalized_odds_difference_overall"] = equalized_odds_difference(df.y_true, df.y_pred, sensitive_features=df[sensitive])
    return out


def main():
    df = pd.read_csv(INPUT)
    df["age_group"] = pd.cut(df.age, bins=[17, 29, 44, 59, float("inf")], labels=["18-29", "30-44", "45-59", "60+"], include_lowest=True).astype(str)
    result = pd.concat([audit(df, "gender"), audit(df, "age_group")], ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
