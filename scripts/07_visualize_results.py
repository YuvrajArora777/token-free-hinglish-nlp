"""
07_visualize_results_fast.py  
FAST MODE version (10x faster)

Generates:
 - Accuracy vs Noise Levels (95% CI or skip)
 - Macro F1 vs Noise Levels
 - Optional: Confusion Matrices
 - PNG + SVG output

Optimizations:
 - Reduced bootstrap samples to 300
 - Optional CI toggle
 - Optional confusion matrix toggle
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import random

# =====================================================
# FAST MODE CONFIG
# =====================================================

FAST_MODE = True                 # Entire fast-mode toggle
USE_CI = True                    # Enable confidence intervals?
CONFUSION_MATRICES = False       # Turn off to save time

BOOTSTRAP_SAMPLES = 300          # MUCH faster than 2000

# =====================================================
# Directories
# =====================================================

PRED_DIR = Path("results/predictions")
SAVE_DIR = Path("results/charts_fast")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

# Matplotlib settings
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.3

# =====================================================
# Helpers
# =====================================================

def load_jsonl(path):
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]

def compute_metrics(records):
    gold = [r["gold"] for r in records]
    pred = [r["pred"] for r in records]
    acc = accuracy_score(gold, pred) * 100
    f1 = f1_score(gold, pred, average="macro") * 100
    return acc, f1, gold, pred

def acc_metric(records):
    acc, _, _, _ = compute_metrics(records)
    return acc

def f1_metric(records):
    _, f1, _, _ = compute_metrics(records)
    return f1

def bootstrap_ci(records, metric_fn, samples=BOOTSTRAP_SAMPLES):
    if not USE_CI:
        # No CI → just return (val,val,val)
        m = metric_fn(records)
        return m, m, m

    n = len(records)
    vals = []
    for _ in range(samples):
        idx = np.random.choice(n, n, replace=True)
        subset = [records[i] for i in idx]
        vals.append(metric_fn(subset))

    vals = np.array(vals)
    return vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


# =====================================================
# Load all datasets + models
# =====================================================

datasets = sorted([d.name for d in PRED_DIR.iterdir() if d.is_dir()])
models = set()

for d in datasets:
    for fp in (PRED_DIR / d).glob("*.jsonl"):
        models.add(fp.stem)

models = sorted(models)

ordered_ds = ["clean", "noisy_low", "noisy_medium", "noisy_high"]
ordered_ds = [d for d in ordered_ds if d in datasets]

# metrics[model][dataset]
metrics = {m: {} for m in models}

print("FAST MODE:", FAST_MODE)
print("Confidence Intervals:", "ON" if USE_CI else "OFF")
print("Confusion Matrices:", "ON" if CONFUSION_MATRICES else "OFF")

for m in models:
    for ds in ordered_ds:
        fp = PRED_DIR / ds / f"{m}.jsonl"
        if not fp.exists():
            continue

        records = load_jsonl(fp)
        acc, f1, gold, pred = compute_metrics(records)

        acc_mean, acc_lo, acc_hi = bootstrap_ci(records, acc_metric)
        f1_mean, f1_lo, f1_hi = bootstrap_ci(records, f1_metric)

        metrics[m][ds] = {
            "records": records,
            "acc": acc,
            "f1": f1,
            "acc_ci": (acc_lo, acc_hi),
            "f1_ci": (f1_lo, f1_hi)
        }

# =====================================================
# Plot: Accuracy vs Noise
# =====================================================

plt.figure(figsize=(10, 6))
x = np.arange(len(models))
w = 0.18

for i, ds in enumerate(ordered_ds):
    vals = [metrics[m][ds]["acc"] for m in models]
    lohi = [metrics[m][ds]["acc_ci"] for m in models]
    yerr = np.array([[vals[j] - lohi[j][0], lohi[j][1] - vals[j]]
                     for j in range(len(models))]).T if USE_CI else None

    plt.bar(x + (i - 1.5) * w, vals, width=w,
            label=ds.replace("noisy_", "noisy "),
            yerr=yerr, capsize=4 if USE_CI else 0)

plt.xticks(x, models)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Noise Levels (FAST MODE)")
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / "accuracy_noise_levels.png")
plt.savefig(SAVE_DIR / "accuracy_noise_levels.svg")
plt.show()

# =====================================================
# Plot: Macro F1 vs Noise
# =====================================================

plt.figure(figsize=(10, 6))

for i, ds in enumerate(ordered_ds):
    vals = [metrics[m][ds]["f1"] for m in models]
    lohi = [metrics[m][ds]["f1_ci"] for m in models]
    yerr = np.array([[vals[j] - lohi[j][0], lohi[j][1] - vals[j]]
                     for j in range(len(models))]).T if USE_CI else None

    plt.bar(x + (i - 1.5) * w, vals, width=w,
            label=ds.replace("noisy_", "noisy "),
            yerr=yerr, capsize=4 if USE_CI else 0)

plt.xticks(x, models)
plt.ylabel("Macro F1 (%)")
plt.title("Macro F1 vs Noise Levels (FAST MODE)")
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / "f1_noise_levels.png")
plt.savefig(SAVE_DIR / "f1_noise_levels.svg")
plt.show()

# =====================================================
# Confusion Matrices (optional)
# =====================================================

if CONFUSION_MATRICES:
    print("Generating confusion matrices (slow)...")

    for m in models:
        for ds in ordered_ds:
            recs = metrics[m][ds]["records"]
            _, _, gold, pred = compute_metrics(recs)

            labels = sorted(list(set(gold + pred)))
            cm = confusion_matrix(gold, pred, labels=labels)
            cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, None] + 1e-12)

            plt.figure(figsize=(8, 6))
            plt.imshow(cm_norm, cmap="Blues")
            plt.colorbar()

            plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
            plt.yticks(range(len(labels)), labels)

            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, f"{cm_norm[i,j]:.2f}",
                             ha='center', va='center')

            plt.title(f"Confusion Matrix: {m} on {ds}")
            plt.xlabel("Predicted")
            plt.ylabel("Gold")

            fname = SAVE_DIR / f"cm_{m}_{ds}"
            plt.tight_layout()
            plt.savefig(fname.with_suffix(".png"))
            plt.savefig(fname.with_suffix(".svg"))
            plt.close()

print("\n⚡ FAST MODE COMPLETED!")
print("Charts saved to:", SAVE_DIR)
