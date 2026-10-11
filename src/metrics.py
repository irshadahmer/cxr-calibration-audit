import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

def compute_ece(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_confidence = np.mean(y_prob[in_bin])
            avg_accuracy = np.mean(y_true[in_bin])
            ece += (bin_size / n_samples) * np.abs(avg_confidence - avg_accuracy)

    return float(ece)

def evaluate_predictions(y_true, y_prob):
    """Calculates AUROC, ECE, and Brier Score."""
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
        
    ece = compute_ece(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)
    return {"auc": auc, "ece": ece, "brier": brier}
