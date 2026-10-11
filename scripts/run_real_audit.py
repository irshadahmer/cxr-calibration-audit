import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchxrayvision as xrv
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from src.metrics import compute_ece, brier_score_loss
from src.recalibration import fit_temperature

def bootstrap_ci(metric_fn, y_true, y_prob, n_boot=500, ci=0.95):
    scores = []
    rng = np.random.RandomState(42)
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    alpha = (1.0 - ci) / 2.0
    return float(np.mean(scores)), float(np.percentile(scores, alpha * 100)), float(np.percentile(scores, (1.0 - alpha) * 100))

def main():
    print("Loading Stanford CheXpert DenseNet-121 (densenet121-res224-chex)...")
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.eval()

    img_dir = "data/external_cxr/images"
    csv_path = "data/external_cxr/metadata.csv"
    ds = xrv.datasets.COVID19_Dataset(imgpath=img_dir, csvpath=csv_path, views=["PA", "AP", "AP Supine"])

    target_task = "Pneumonia"
    model_task_idx = model.pathologies.index(target_task)
    ds_task_idx = ds.pathologies.index(target_task)

    logits_list = []
    labels_list = []
    ages_list = []
    sexes_list = []

    print(f"Running inference on {len(ds)} real external clinical radiographs...")

    with torch.no_grad():
        for i in tqdm(range(len(ds))):
            sample = ds[i]
            label = sample["lab"][ds_task_idx]
            if np.isnan(label):
                continue

            # Normalized tensor: sample["img"] is (1, H, W) in [-1024, 1024]
            img_tensor = torch.from_numpy(sample["img"]).float().unsqueeze(0) # (1, 1, H, W)
            # Bilinear interpolation to standard 224x224
            img_224 = F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)

            out = model(img_224)
            logit = out[0, model_task_idx].item()

            logits_list.append(logit)
            labels_list.append(int(label > 0.5))

            row = ds.csv.iloc[i]
            ages_list.append(row.get('age', np.nan))
            sexes_list.append(row.get('sex', np.nan))

    logits = np.array(logits_list)
    y_true = np.array(labels_list)
    probs = 1.0 / (1.0 + np.exp(-logits))
    ages = pd.to_numeric(pd.Series(ages_list), errors='coerce').values
    sexes = np.array(sexes_list)

    print(f"\nExtracted {len(y_true)} valid cases. Target: {target_task} (Prevalence: {np.mean(y_true):.3f})")

    # 1. External Cohort Baseline
    auc = roc_auc_score(y_true, probs)
    ece_m, ece_lo, ece_hi = bootstrap_ci(lambda yt, yp: compute_ece(yt, yp), y_true, probs)
    br_m, br_lo, br_hi = bootstrap_ci(lambda yt, yp: brier_score_loss(yt, yp), y_true, probs)

    print("\n" + "="*50)
    print("REAL CLINICAL AUDIT: UNCALIBRATED EXTERNAL SITE")
    print("="*50)
    print(f"AUROC: {auc:.4f}")
    print(f"ECE:   {ece_m:.4f} [95% CI: {ece_lo:.4f} - {ece_hi:.4f}]")
    print(f"Brier: {br_m:.4f} [95% CI: {br_lo:.4f} - {br_hi:.4f}]")

    # 2. Recalibration Sample Efficiency
    budgets = [25, 50, 100, 200]
    results = []

    for b in budgets:
        b_eces = []
        b_temps = []
        for seed in range(100):
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(y_true))
            cal_idx, test_idx = perm[:b], perm[b:]

            if len(np.unique(y_true[cal_idx])) < 2:
                continue

            T, _ = fit_temperature(logits[cal_idx], y_true[cal_idx])
            prob_test_recal = 1.0 / (1.0 + np.exp(-(logits[test_idx] / T)))
            b_eces.append(compute_ece(y_true[test_idx], prob_test_recal))
            b_temps.append(T)

        results.append({
            "Audit Budget": b,
            "Mean T": float(np.mean(b_temps)),
            "Mean ECE": float(np.mean(b_eces)),
            "ECE 2.5%": float(np.percentile(b_eces, 2.5)),
            "ECE 97.5%": float(np.percentile(b_eces, 97.5))
        })

    res_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("RECALIBRATION SAMPLE-EFFICIENCY (100 Monte Carlo Iterations)")
    print("="*50)
    print(res_df.to_string(index=False))

    # 3. Subgroup Demographic Calibration Audit
    print("\n" + "="*50)
    print("DEMOGRAPHIC FAIRNESS & SUBGROUP CALIBRATION")
    print("="*50)

    for s in ['M', 'F']:
        mask = (sexes == s)
        if np.sum(mask) > 10 and len(np.unique(y_true[mask])) > 1:
            s_auc = roc_auc_score(y_true[mask], probs[mask])
            s_ece = compute_ece(y_true[mask], probs[mask])
            s_brier = brier_score_loss(y_true[mask], probs[mask])
            print(f"Sex '{s}' (N={np.sum(mask)}): AUROC={s_auc:.3f} | ECE={s_ece:.4f} | Brier={s_brier:.4f}")

    valid_age = ~np.isnan(ages)
    young_mask = valid_age & (ages < 55)
    older_mask = valid_age & (ages >= 55)
    if np.sum(young_mask) > 10 and len(np.unique(y_true[young_mask])) > 1:
        print(f"Age < 55 (N={np.sum(young_mask)}): AUROC={roc_auc_score(y_true[young_mask], probs[young_mask]):.3f} | ECE={compute_ece(y_true[young_mask], probs[young_mask]):.4f} | Brier={brier_score_loss(y_true[young_mask], probs[young_mask]):.4f}")
    if np.sum(older_mask) > 10 and len(np.unique(y_true[older_mask])) > 1:
        print(f"Age >= 55 (N={np.sum(older_mask)}): AUROC={roc_auc_score(y_true[older_mask], probs[older_mask]):.3f} | ECE={compute_ece(y_true[older_mask], probs[older_mask]):.4f} | Brier={brier_score_loss(y_true[older_mask], probs[older_mask]):.4f}")

    os.makedirs("results", exist_ok=True)
    res_df.to_csv("results/real_audit_benchmark.csv", index=False)

    case_df = pd.DataFrame({
        "logit": logits,
        "prob_uncal": probs,
        "y_true": y_true,
        "age": ages,
        "sex": sexes
    })
    case_df.to_csv("results/real_predictions.csv", index=False)
    print("\nSaved genuine outputs to results/real_audit_benchmark.csv and results/real_predictions.csv")

if __name__ == '__main__':
    main()
