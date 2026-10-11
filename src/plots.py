import numpy as np
import matplotlib.pyplot as plt

def plot_reliability_diagrams(y_true, probs_dict, save_path="figures/fig1_reliability_diagram.png", n_bins=10):
    fig, axes = plt.subplots(1, len(probs_dict), figsize=(5 * len(probs_dict), 4.5), sharey=True)
    if len(probs_dict) == 1:
        axes = [axes]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for ax, (title, y_prob) in zip(axes, probs_dict.items()):
        accs = []
        confs = []
        for i in range(n_bins):
            in_bin = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i+1]) if i > 0 else (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i+1])
            if np.sum(in_bin) > 0:
                accs.append(np.mean(y_true[in_bin]))
                confs.append(np.mean(y_prob[in_bin]))
            else:
                accs.append(np.nan)
                confs.append(centers[i])

        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
        ax.plot(confs, accs, marker="o", linewidth=2, label="Model Output", color="#1f77b4")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax == axes[0]:
            ax.set_ylabel("True Fraction of Positives")
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_recalibration_curve(budgets, ece_scores, save_path="figures/fig2_recovery_curve.png"):
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(budgets, ece_scores, marker="s", color="#d62728", linewidth=2.5, markersize=7)
    plt.axhline(y=ece_scores[0], color="grey", linestyle="--", label="Uncalibrated External Site")
    plt.title("ECE Recovery vs. Local Audit Budget (N)", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Labeled Scans from New Hospital (N)")
    plt.ylabel("Expected Calibration Error (ECE)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
