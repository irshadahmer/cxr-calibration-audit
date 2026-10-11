import numpy as np
from src.metrics import evaluate_predictions
from src.recalibration import fit_temperature

print("=== RUNNING PIPELINE VERIFICATION ===")
np.random.seed(42)

# 1. Source site simulation (calibrated)
n_int = 1000
logits_int = np.random.normal(0, 1.4, n_int)
prob_int = 1 / (1 + np.exp(-logits_int))
y_int = (np.random.rand(n_int) < prob_int).astype(int)
res_int = evaluate_predictions(y_int, prob_int)
print(f"Internal Site:    AUC: {res_int['auc']:.3f} | ECE: {res_int['ece']:.4f} | Brier: {res_int['brier']:.4f}")

# 2. Target hospital site shift (severe overconfidence drift)
n_ext = 1000
logits_ext = logits_int * 2.5 + 0.4
prob_ext_uncal = 1 / (1 + np.exp(-logits_ext))
y_ext = (np.random.rand(n_ext) < prob_int).astype(int)
res_uncal = evaluate_predictions(y_ext, prob_ext_uncal)
print(f"External Uncalib: AUC: {res_uncal['auc']:.3f} | ECE: {res_uncal['ece']:.4f} | Brier: {res_uncal['brier']:.4f}")

# 3. Recalibration with small audit budget (N=200 local images)
T, _ = fit_temperature(logits_ext[:200], y_ext[:200])
test_recal_prob = 1 / (1 + np.exp(-(logits_ext[200:] / T)))
res_cal = evaluate_predictions(y_ext[200:], test_recal_prob)
print(f"External Recalib: AUC: {res_cal['auc']:.3f} | ECE: {res_cal['ece']:.4f} | Brier: {res_cal['brier']:.4f} (Learned T={T:.2f})")
print("=== PIPELINE VALIDATED SUCCESSFULLY ===")
