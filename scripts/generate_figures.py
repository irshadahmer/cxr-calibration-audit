import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300
})

os.makedirs("results/figures", exist_ok=True)

df = pd.read_csv("results/real_predictions_calibrated.csv")
bench_df = pd.read_csv("results/platt_vs_temperature_benchmark.csv")
y_true = df["y_true"].values

def compute_calibration_curve(y_true, probs, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    true_fractions = []
    mean_predicted = []
    counts = []

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        counts.append(np.sum(mask))
        if np.sum(mask) > 0:
            true_fractions.append(np.mean(y_true[mask]))
            mean_predicted.append(np.mean(probs[mask]))
        else:
            true_fractions.append(np.nan)
            mean_predicted.append(bin_centers[i])

    return np.array(mean_predicted), np.array(true_fractions), np.array(counts), bins

# ==========================================================
# FIGURE 1: Reliability Diagrams & Confidence Histograms
# ==========================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

configs = [
    ("Uncalibrated Baseline", df["prob_uncal"].values, "#d95f02"),
    ("Temperature Scaling (T=1.46)", df["prob_ts"].values, "#7570b3"),
    ("Platt Scaling (w*z + b)", df["prob_platt"].values, "#1b9e77")
]

for col_idx, (title, probs, color) in enumerate(configs):
    ax_top = axes[0, col_idx]
    ax_bottom = axes[1, col_idx]

    pred_f, true_f, counts, bins = compute_calibration_curve(y_true, probs, n_bins=10)

    # Top panel: Reliability curve
    ax_top.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7, linewidth=1.2)
    valid = ~np.isnan(true_f)
    ax_top.plot(pred_f[valid], true_f[valid], marker="s", markersize=6, color=color, linewidth=2, label="Observed Rate")
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.set_title(title, weight="bold")
    ax_top.set_xlabel("Mean Predicted Risk")
    if col_idx == 0:
        ax_top.set_ylabel("Empirical True Fraction")
    ax_top.legend(loc="upper left", frameon=True, fontsize=8)
    ax_top.grid(True, linestyle=":", alpha=0.6)

    # Bottom panel: Confidence histogram
    ax_bottom.hist(probs, bins=bins, color=color, edgecolor="black", alpha=0.75, rwidth=0.85)
    ax_bottom.set_xlim(0, 1)
    ax_bottom.set_xlabel("Predicted Risk Bin")
    if col_idx == 0:
        ax_bottom.set_ylabel("Count")
    ax_bottom.grid(True, linestyle=":", alpha=0.4)

plt.suptitle("External Reliability Audit: Stanford CheXpert Model on Real Cohort (N=697)", weight="bold", y=0.98)
plt.tight_layout()
plt.savefig("results/figures/fig1_calibration_comparison.png")
plt.close()
print("Saved results/figures/fig1_calibration_comparison.png")

# ==========================================================
# FIGURE 2: Recalibration Sample Efficiency
# ==========================================================
fig, ax = plt.subplots(figsize=(8, 5))

budgets = bench_df["Budget"].values
ts_ece = bench_df["TS ECE (Mean)"].values
platt_ece = bench_df["Platt ECE (Mean)"].values

# Extract Platt CI bounds from formatted strings
ci_lo = [float(s.strip("[]").split(",")[0]) for s in bench_df["Platt ECE [2.5%, 97.5%]"]]
ci_hi = [float(s.strip("[]").split(",")[1]) for s in bench_df["Platt ECE [2.5%, 97.5%]"]]

ax.axhline(0.3486, color="#d95f02", linestyle="--", linewidth=1.5, label="Uncalibrated Baseline (ECE = 0.349)")
ax.plot(budgets, ts_ece, marker="o", color="#7570b3", linewidth=2, label="Temperature Scaling (1-parameter)")
ax.plot(budgets, platt_ece, marker="s", color="#1b9e77", linewidth=2.5, label="Platt Scaling (w*z + b, 2-parameter)")
ax.fill_between(budgets, ci_lo, ci_hi, color="#1b9e77", alpha=0.2, label="Platt 95% Bootstrap CI")

ax.set_title("Recalibration Sample Efficiency (100 Monte Carlo Iterations)", weight="bold")
ax.set_xlabel("Clinical Audit Budget (Scans Sampled)")
ax.set_ylabel("Expected Calibration Error (ECE)")
ax.set_xticks(budgets)
ax.set_ylim(-0.02, 0.45)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="center right", frameon=True)

plt.tight_layout()
plt.savefig("results/figures/fig2_sample_efficiency.png")
plt.close()
print("Saved results/figures/fig2_sample_efficiency.png")

# ==========================================================
# FIGURE 3: Demographic Subgroup Calibration & ROC Curves
# ==========================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# Panel A: ROC by Sex
ax_a = axes[0, 0]
for s, col in [("M", "#2b83ba"), ("F", "#d7191c")]:
    mask = (df["sex"] == s).values
    fpr, tpr, _ = roc_curve(y_true[mask], df["prob_uncal"].values[mask])
    auc = roc_auc_score(y_true[mask], df["prob_uncal"].values[mask])
    ax_a.plot(fpr, tpr, label=f"Sex {s} (AUC = {auc:.3f}, N={np.sum(mask)})", color=col, linewidth=2)
ax_a.plot([0, 1], [0, 1], "k:", alpha=0.5)
ax_a.set_title("ROC Curves by Sex", weight="bold")
ax_a.set_xlabel("False Positive Rate")
ax_a.set_ylabel("True Positive Rate")
ax_a.legend(loc="lower right", frameon=True)
ax_a.grid(True, linestyle=":", alpha=0.6)

# Panel B: ROC by Age
ax_b = axes[0, 1]
valid_age = ~df["age"].isna().values
age_val = df["age"].values
young = valid_age & (age_val < 55)
older = valid_age & (age_val >= 55)

for mask, lab, col in [(young, "Age < 55", "#fdae61"), (older, "Age >= 55", "#2c7bb6")]:
    fpr, tpr, _ = roc_curve(y_true[mask], df["prob_uncal"].values[mask])
    auc = roc_auc_score(y_true[mask], df["prob_uncal"].values[mask])
    ax_b.plot(fpr, tpr, label=f"{lab} (AUC = {auc:.3f}, N={np.sum(mask)})", color=col, linewidth=2)
ax_b.plot([0, 1], [0, 1], "k:", alpha=0.5)
ax_b.set_title("ROC Curves by Age Group", weight="bold")
ax_b.set_xlabel("False Positive Rate")
ax_b.set_ylabel("True Positive Rate")
ax_b.legend(loc="lower right", frameon=True)
ax_b.grid(True, linestyle=":", alpha=0.6)

# Panel C: Reliability Curves by Sex (Platt Calibrated)
ax_c = axes[1, 0]
ax_c.plot([0, 1], [0, 1], "k--", alpha=0.6)
for s, col in [("M", "#2b83ba"), ("F", "#d7191c")]:
    mask = (df["sex"] == s).values
    pred_f, true_f, _, _ = compute_calibration_curve(y_true[mask], df["prob_platt"].values[mask], n_bins=8)
    valid = ~np.isnan(true_f)
    ax_c.plot(pred_f[valid], true_f[valid], marker="o", label=f"Sex {s} (Platt)", color=col, linewidth=1.8)
ax_c.set_title("Calibrated Reliability by Sex", weight="bold")
ax_c.set_xlabel("Mean Predicted Risk")
ax_c.set_ylabel("Empirical True Fraction")
ax_c.legend(loc="upper left", frameon=True)
ax_c.grid(True, linestyle=":", alpha=0.6)

# Panel D: Reliability Curves by Age (Platt Calibrated)
ax_d = axes[1, 1]
ax_d.plot([0, 1], [0, 1], "k--", alpha=0.6)
for mask, lab, col in [(young, "Age < 55", "#fdae61"), (older, "Age >= 55", "#2c7bb6")]:
    pred_f, true_f, _, _ = compute_calibration_curve(y_true[mask], df["prob_platt"].values[mask], n_bins=8)
    valid = ~np.isnan(true_f)
    ax_d.plot(pred_f[valid], true_f[valid], marker="s", label=f"{lab} (Platt)", color=col, linewidth=1.8)
ax_d.set_title("Calibrated Reliability by Age", weight="bold")
ax_d.set_xlabel("Mean Predicted Risk")
ax_d.set_ylabel("Empirical True Fraction")
ax_d.legend(loc="upper left", frameon=True)
ax_d.grid(True, linestyle=":", alpha=0.6)

plt.suptitle("Demographic Fairness & Subgroup Performance Audit", weight="bold", y=0.98)
plt.tight_layout()
plt.savefig("results/figures/fig3_demographic_fairness.png")
plt.close()
print("Saved results/figures/fig3_demographic_fairness.png")
