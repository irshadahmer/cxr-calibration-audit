import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.metrics import compute_ece, brier_score_loss
from src.recalibration import fit_temperature

def main():
    df = pd.read_csv("results/real_predictions.csv")
    logits = df["logit"].values
    probs_uncal = df["prob_uncal"].values
    y_true = df["y_true"].values
    n = len(y_true)

    print(f"Loaded {n} patient predictions. Cohort prevalence: {np.mean(y_true):.3f}")

    # Baseline performance
    base_auc = roc_auc_score(y_true, probs_uncal)
    base_ece = compute_ece(y_true, probs_uncal)
    base_brier = brier_score_loss(y_true, probs_uncal)

    print("\n" + "="*58)
    print("RECALIBRATION COMPARISON: TEMPERATURE SCALING VS PLATT")
    print("="*58)
    print(f"Baseline (Uncalibrated): AUROC={base_auc:.4f} | ECE={base_ece:.4f} | Brier={base_brier:.4f}")

    budgets = [25, 50, 100, 200]
    n_splits = 100
    rows = []

    for b in budgets:
        ts_eces, ts_briers = [], []
        platt_eces, platt_briers = [], []
        platt_slopes, platt_intercepts = [], []

        for seed in range(n_splits):
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n)
            cal_idx, test_idx = perm[:b], perm[b:]

            y_cal, y_test = y_true[cal_idx], y_true[test_idx]
            z_cal, z_test = logits[cal_idx], logits[test_idx]

            # Require both classes in calibration set
            if len(np.unique(y_cal)) < 2:
                continue

            # 1. Temperature Scaling (1-parameter: z / T)
            T, _ = fit_temperature(z_cal, y_cal)
            prob_test_ts = 1.0 / (1.0 + np.exp(-(z_test / T)))
            ts_eces.append(compute_ece(y_test, prob_test_ts))
            ts_briers.append(brier_score_loss(y_test, prob_test_ts))

            # 2. Platt Scaling (2-parameter: w * z + b)
            lr = LogisticRegression(solver="lbfgs", C=1.0)
            lr.fit(z_cal.reshape(-1, 1), y_cal)
            prob_test_platt = lr.predict_proba(z_test.reshape(-1, 1))[:, 1]
            platt_eces.append(compute_ece(y_test, prob_test_platt))
            platt_briers.append(brier_score_loss(y_test, prob_test_platt))
            platt_slopes.append(lr.coef_[0][0])
            platt_intercepts.append(lr.intercept_[0])

        rows.append({
            "Budget": b,
            "TS ECE (Mean)": np.mean(ts_eces),
            "Platt ECE (Mean)": np.mean(platt_eces),
            "Platt ECE [2.5%, 97.5%]": f"[{np.percentile(platt_eces, 2.5):.4f}, {np.percentile(platt_eces, 97.5):.4f}]",
            "Platt Brier": np.mean(platt_briers),
            "Mean Slope (w)": np.mean(platt_slopes),
            "Mean Intercept (b)": np.mean(platt_intercepts),
        })

    res_df = pd.DataFrame(rows)
    print("\n" + res_df.to_string(index=False))
    res_df.to_csv("results/platt_vs_temperature_benchmark.csv", index=False)
    print("\nSaved benchmark to results/platt_vs_temperature_benchmark.csv")

    # Fit final full Platt model on external cohort to store calibrated probabilities for plotting
    lr_full = LogisticRegression(solver="lbfgs", C=1.0)
    lr_full.fit(logits.reshape(-1, 1), y_true)
    df["prob_platt"] = lr_full.predict_proba(logits.reshape(-1, 1))[:, 1]
    T_full, _ = fit_temperature(logits, y_true)
    df["prob_ts"] = 1.0 / (1.0 + np.exp(-(logits / T_full)))
    df.to_csv("results/real_predictions_calibrated.csv", index=False)
    print("Saved calibrated predictions to results/real_predictions_calibrated.csv")

if __name__ == '__main__':
    main()
