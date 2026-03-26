"""
09_visualize_all.py
Unified Visualization Script (All Graphs)
 - Accuracy vs Noise Levels
 - Macro F1 vs Noise Levels
 - NRI Line Charts
 - Full Confusion Matrices
 - Organized folder structure
 - PNG + SVG outputs

FAST MODE:
 - Bootstrap CI = 300 samples
 - Matplotlib-only
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import random

# ============================================================================
# FAST MODE SETTINGS
# ============================================================================
FAST_MODE = True
BOOTSTRAP_SAMPLES = 300          # Much faster than 2000
USE_CI = True                    # Toggle confidence intervals
GENERATE_CM = True               # Generate Confusion Matrices

# ============================================================================
# PATHS
# ============================================================================
PRED_DIR = Path("results/predictions")
BASE_OUT = Path("results/charts")
OUT_ACC = BASE_OUT / "accuracy"
OUT_F1 = BASE_OUT / "f1"
OUT_NRI = BASE_OUT / "nri"
OUT_CM = BASE_OUT / "confusion"

for p in [OUT_ACC, OUT_F1, OUT_NRI, OUT_CM]:
    p.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MATPLOTLIB STYLE
# ============================================================================
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.3

random.seed(42)
np.random.seed(42)

# ============================================================================
# HELPERS
# ============================================================================
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

def bootstrap_ci(records, fn, n=BOOTSTRAP_SAMPLES):
    if not USE_CI:
        m = fn(records)
        return m, m, m

    vals = []
    N = len(records)

    for _ in range(n):
        idx = np.random.choice(N, N, replace=True)
        vals.append(fn([records[i] for i in idx]))

    vals = np.array(vals)
    return vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

# ============================================================================
# LOAD MODELS + DATASETS
# ============================================================================
datasets = sorted([d.name for d in PRED_DIR.iterdir() if d.is_dir()])
models = sorted({fp.stem for ds in datasets for fp in (PRED_DIR / ds).glob("*.jsonl")})

ordered_ds = ["clean", "noisy_low", "noisy_medium", "noisy_high"]
ordered_ds = [d for d in ordered_ds if d in datasets]

metrics = {m: {} for m in models}

# ============================================================================
# READ PREDICTIONS + METRICS
# ============================================================================
print("Loading predictions and computing metrics...")

for m in models:
    for ds in ordered_ds:
        fp = PRED_DIR / ds / f"{m}.jsonl"
        if not fp.exists(): 
            continue

        recs = load_jsonl(fp)
        acc, f1, gold, pred = compute_metrics(recs)

        acc_mean, acc_lo, acc_hi = bootstrap_ci(recs, acc_metric)
        f1_mean, f1_lo, f1_hi = bootstrap_ci(recs, f1_metric)

        metrics[m][ds] = {
            "records": recs,
            "acc": acc,
            "f1": f1,
            "acc_ci": (acc_lo, acc_hi),
            "f1_ci": (f1_lo, f1_hi)
        }

# ============================================================================
# 1. ACCURACY VS NOISE
# ============================================================================
print("Generating Accuracy plots...")

x = np.arange(len(models))
w = 0.18

plt.figure(figsize=(10, 6))
for i, ds in enumerate(ordered_ds):
    vals = [metrics[m][ds]["acc"] for m in models]
    lohi = [metrics[m][ds]["acc_ci"] for m in models]

    yerr = np.array([
        [vals[j] - lohi[j][0], lohi[j][1] - vals[j]]
        for j in range(len(models))
    ]).T if USE_CI else None

    plt.bar(
        x + (i - 1.5) * w, vals, width=w,
        label=ds.replace("noisy_", "noisy "),
        yerr=yerr, capsize=4 if USE_CI else 0
    )

plt.xticks(x, models)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Noise Levels")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ACC / "accuracy_vs_noise.png")
plt.savefig(OUT_ACC / "accuracy_vs_noise.svg")
plt.show()

# ============================================================================
# 2. MACRO F1 VS NOISE
# ============================================================================
print("Generating Macro F1 plots...")

plt.figure(figsize=(10, 6))
for i, ds in enumerate(ordered_ds):
    vals = [metrics[m][ds]["f1"] for m in models]
    lohi = [metrics[m][ds]["f1_ci"] for m in models]

    yerr = np.array([
        [vals[j] - lohi[j][0], lohi[j][1] - vals[j]]
        for j in range(len(models))
    ]).T if USE_CI else None

    plt.bar(
        x + (i - 1.5) * w, vals, width=w,
        label=ds.replace("noisy_", "noisy "),
        yerr=yerr, capsize=4 if USE_CI else 0
    )

plt.xticks(x, models)
plt.ylabel("Macro F1 (%)")
plt.title("Macro F1 vs Noise Levels")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_F1 / "f1_vs_noise.png")
plt.savefig(OUT_F1 / "f1_vs_noise.svg")
plt.show()

# ============================================================================
# 3. NRI PLOT
# ============================================================================
print("Generating NRI plot...")

plt.figure(figsize=(10, 6))

for i, noise_level in enumerate(["noisy_low", "noisy_medium", "noisy_high"]):
    vals = []
    for m in models:
        clean_acc = metrics[m]["clean"]["acc"]
        noisy_acc = metrics[m][noise_level]["acc"]
        vals.append(noisy_acc / clean_acc * 100)

    plt.plot(models, vals, marker="o", label=noise_level.replace("_", " ").title())

plt.ylabel("NRI (%)")
plt.title("Noise Robustness Index Across Levels")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_NRI / "nri_plot.png")
plt.savefig(OUT_NRI / "nri_plot.svg")
plt.show()

# ============================================================================
# 4. CONFUSION MATRICES
# ============================================================================
if GENERATE_CM:
    print("Generating Confusion Matrices...")

    for m in models:
        for ds in ordered_ds:
            recs = metrics[m][ds]["records"]
            acc, f1, gold, pred = compute_metrics(recs)

            labels = sorted(list(set(gold + pred)))
            cm = confusion_matrix(gold, pred, labels=labels)
            cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, None] + 1e-12)

            plt.figure(figsize=(8, 6))
            plt.imshow(cm_norm, cmap="Blues")
            plt.colorbar()

            plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
            plt.yticks(range(len(labels)), labels)

            # Annotate each cell
            for i in range(len(labels)):
                for j in range(len(labels)):
                    v = cm_norm[i, j]
                    c = "white" if v > 0.5 else "black"
                    plt.text(j, i, f"{v*100:.1f}%", ha="center", va="center", color=c)

            plt.title(f"{m.upper()} — {ds.replace('_', ' ').title()}")
            plt.xlabel("Predicted")
            plt.ylabel("True")

            fname = OUT_CM / f"cm_{m}_{ds}"
            plt.tight_layout()
            plt.savefig(fname.with_suffix(".png"))
            plt.savefig(fname.with_suffix(".svg"))
            plt.close()

print("\n🎉 ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print(f"Saved under: {BASE_OUT}")
