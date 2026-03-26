"""
08_evaluate_with_predictions.py
Evaluates:
 - mBERT
 - CANINE
 - ByT5 (custom classifier)
Saves:
 - per-sample predictions (JSONL)
 - gold labels
 - logits/probabilities
Enables:
 - Macro F1
 - Confusion Matrices
 - Confidence Intervals (bootstrap)
"""

import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)

# Custom ByT5Classifier
from scripts.byt5_model import ByT5Classifier  # you created this file earlier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Datasets
DATASETS = {
    "clean": "data/human_annotated/test.tsv",
    "noisy_low": "data/noisy_humanlike_low/test.tsv",
    "noisy_medium": "data/noisy_humanlike_medium/test.tsv",
    "noisy_high": "data/noisy_humanlike_high/test.tsv",
}

# Models
MODELS = {
    "mbert": "models/mbert/final",
    "canine": "models/canine/final",
    "byt5": "models/byt5/final"
}

# Output
OUT_DIR = Path("results/predictions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------
# Helper: Save predictions for CI + CM + F1
# -------------------------------------------
def save_predictions(model_name, dataset_name, rows, golds, preds, probs):
    out_path = OUT_DIR / dataset_name
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = out_path / f"{model_name}.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            record = {
                "id": i,
                "text": row["cs_query"],
                "gold": golds[i],
                "pred": preds[i],
                "probs": probs[i],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved predictions → {file_path}")


# -------------------------------------------
# Evaluate mBERT & CANINE
# -------------------------------------------
def evaluate_subword(model_name, model_path, dataset):
    print(f"→ Evaluating {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()

    label2id = model.config.label2id
    id2label = {v: k for k, v in label2id.items()}

    rows, preds, probs, golds = [], [], [], []

    for row in dataset:
        text = row["cs_query"]
        gold = row["domain"].lower()

        rows.append(row)
        golds.append(gold)

        enc = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)

        with torch.no_grad():
            out = model(**enc)
            log = out.logits[0]
            prob = F.softmax(log, dim=-1).cpu().numpy()
            pred_id = int(log.argmax().cpu().item())
            pred_label = id2label[pred_id]

        preds.append(pred_label)
        probs.append(prob.tolist())

    return rows, golds, preds, probs


# -------------------------------------------
# Evaluate ByT5 (encoder-only)
# -------------------------------------------
def evaluate_byt5(model_path, dataset):
    print("→ Evaluating ByT5")

    config = AutoConfig.from_pretrained(model_path)
    label2id = config.label2id
    id2label = {v: k for k, v in label2id.items()}

    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = ByT5Classifier("google/byt5-small", num_labels)
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    rows, preds, probs, golds = [], [], [], []

    for row in dataset:
        text = row["cs_query"]
        gold = row["domain"].lower()

        rows.append(row)
        golds.append(gold)

        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)

        with torch.no_grad():
            out = model(**enc)
            logits = out["logits"][0]
            prob = F.softmax(logits, dim=-1).cpu().numpy()
            pred_id = int(prob.argmax())
            pred_label = id2label[pred_id]

        preds.append(pred_label)
        probs.append(prob.tolist())

    return rows, golds, preds, probs


# -------------------------------------------
# MAIN EVALUATION LOOP
# -------------------------------------------
if __name__ == "__main__":

    for ds_name, ds_path in DATASETS.items():

        print("\n===============================================")
        print(f"📄 Loading dataset: {ds_name}")
        print("===============================================")

        dataset = load_dataset("csv", data_files={"test": ds_path}, delimiter="\t")["test"]

        for model_name, model_path in MODELS.items():

            print(f"\n🔍 Evaluating {model_name.upper()} on {ds_name.upper()}...")

            if model_name == "byt5":
                rows, golds, preds, probs = evaluate_byt5(model_path, dataset)
            else:
                rows, golds, preds, probs = evaluate_subword(model_name, model_path, dataset)

            # Save predictions → used later for:
            # - Macro F1
            # - Accuracy
            # - Confusion matrices
            # - Confidence intervals (bootstrap)
            save_predictions(model_name, ds_name, rows, golds, preds, probs)

    print("\n======================================================")
    print("✅ All models fully evaluated with per-sample predictions.")
    print("You can now run:  python scripts/07_visualize_results_full.py")
    print("======================================================")
