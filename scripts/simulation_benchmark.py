import numpy as np
import pandas as pd
from src.metrics import evaluate_predictions
from src.recalibration import fit_temperature
from src.plots import plot_reliability_diagrams, plot_recalibration_curve

print("=== STARTING FULL BENCHMARK EXPERIMENT ===")
np.random.seed(101)

# 1. Internal Hospital (Well-calibrated baseline, N=2000)
n_int = 2000
logits_int = np.random.normal(-0.5, 1.2, n_int)
probs_int = 1 / (1 + np.exp(-logits_int))
y_int = (np.random.rand(n_int) < probs_int).astype(int)
res_int = evaluate_predictions(y_int, probs_int)

# 2. External Target Hospital (Under distribution shift, N=3000)
n_ext = 3000
logits_ext = np.random.normal(-0.5, 1.2, n_ext) * 2.1 + 0.35
probs_ext_uncal = 1 / (1 + np.exp(-logits_ext))
true_probs_ext = 1 / (1 + np.exp(-logits_ext / 2.1))
y_ext = (np.random.rand(n_ext) < true_probs_ext).astype(int)
res_ext_uncal = evaluate_predictions(y_ext, probs_ext_uncal)

# 3. Recalibration across budgets: N = [50, 100, 250, 500]
budgets = [50, 100, 250, 500]
results_list = [
    {"Cohort": "Internal Baseline", "Budget_N": 0, "Method": "None", **res_int},
    {"Cohort": "External Uncalibrated", "Budget_N": 0, "Method": "None", **res_ext_uncal}
]

ece_curve = [res_ext_uncal['ece']]
recal_probs = {"External Uncalibrated": probs_ext_uncal[500:]}

for n in budgets:
    T, _ = fit_temperature(logits_ext[:n], y_ext[:n])
    test_logits = logits_ext[500:]
    test_y = y_ext[500:]
    cal_prob = 1 / (1 + np.exp(-(test_logits / T)))
    res_n = evaluate_predictions(test_y, cal_prob)
    results_list.append({"Cohort": "External Recalibrated", "Budget_N": n, "Method": f"Temperature (T={T:.2f})", **res_n})
    ece_curve.append(res_n['ece'])
    if n == 250:
        recal_probs["Recalibrated (N=250)"] = cal_prob

# 4. Save and display results
df_results = pd.DataFrame(results_list)
df_results.to_csv("results/calibration_benchmark.csv", index=False)
print("\n--- BENCHMARK RESULTS TABLE ---")
print(df_results.to_string(index=False))

# 5. Generate Figures
plot_reliability_diagrams(y_ext[500:], recal_probs, save_path="figures/fig1_reliability_diagram.png")
plot_recalibration_curve([0] + budgets, ece_curve, save_path="figures/fig2_recovery_curve.png")
print("\n=== BENCHMARK COMPLETED SUCCESSFULLY ===")
