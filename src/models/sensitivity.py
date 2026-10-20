from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "reports" / "test_predictions.csv"
OUTPUT = ROOT / "reports" / "sensitivity_thresholds.csv"


def main():
    df = pd.read_csv(INPUT)
    rows = []
    for threshold in np.arange(0.30, 0.71, 0.05):
        pred = (df.y_prob >= threshold).astype(int)
        rows.append({
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(df.y_true, pred),
            "precision": precision_score(df.y_true, pred, zero_division=0),
            "recall": recall_score(df.y_true, pred, zero_division=0),
            "f1": f1_score(df.y_true, pred, zero_division=0),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
