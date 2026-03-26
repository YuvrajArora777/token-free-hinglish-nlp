# ============================================================
# 06_evaluate_models.py  (MULTI NOISE TIERS + FIXED BYT5)
# ============================================================

import json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import evaluate
import os, sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Custom ByT5
try:
    from scripts.byt5_model import ByT5Classifier
except Exception:
    try:
        from byt5_model import ByT5Classifier
    except Exception:
        ByT5Classifier = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------
# DATASETS
# ---------------------------------------------------------------------
DATASETS = {
    "clean": "data/human_annotated",
    "noisy_low": "data/noisy_humanlike_low",
    "noisy_medium": "data/noisy_humanlike_medium",
    "noisy_high": "data/noisy_humanlike_high",
}

RESULTS_OUT = "results/metrics/final_nri.json"

MODELS = {
    "mbert": "models/mbert/final",
    "canine": "models/canine/final",
    "byt5": "models/byt5/final"
}

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


# ---------------------------------------------------------------------
# LOAD LABEL MAPPINGS
# ---------------------------------------------------------------------
def load_label_maps(model_path: str):
    id2label = None
    label2id = None
    p = Path(model_path)

    id2 = p / "id2label.json"
    l2 = p / "label2id.json"

    if id2.exists():
        raw = json.load(open(id2, "r"))
        id2label = {}
        for k, v in raw.items():
            try:
                id2label[int(k)] = v
            except:
                id2label[k] = v

    if l2.exists():
        label2id = json.load(open(l2, "r"))

    if id2label is None:
        cfg = AutoConfig.from_pretrained(model_path)
        if getattr(cfg, "id2label", None):
            id2label = {}
            for k, v in cfg.id2label.items():
                try:
                    id2label[int(k)] = v
                except:
                    id2label[k] = v
        if getattr(cfg, "label2id", None):
            label2id = cfg.label2id

    return id2label, label2id


# ---------------------------------------------------------------------
# MBERT / CANINE
# ---------------------------------------------------------------------
def evaluate_mbert_or_canine(model_path, dataset):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()

    pred_ids, gold_ids = [], []

    label2id = model.config.label2id
    if not label2id:
        raise ValueError("No label2id found in config")

    for row in dataset:
        text = row["cs_query"]
        gold = row["domain"]

        tokens = tok(text, return_tensors="pt", truncation=True).to(DEVICE)
        with torch.no_grad():
            out = model(**tokens)
        pred_id = int(out.logits.argmax(-1).item())
        gold_id = label2id.get(gold.lower())

        pred_ids.append(pred_id)
        gold_ids.append(gold_id)

    acc = accuracy_metric.compute(predictions=pred_ids, references=gold_ids)["accuracy"]
    f1s = f1_metric.compute(predictions=pred_ids, references=gold_ids, average="macro")["f1"]
    return acc, f1s


# ---------------------------------------------------------------------
# BYT5 ENCODER-ONLY
# ---------------------------------------------------------------------
def evaluate_byt5(model_path, dataset):
    print(">> Using custom ByT5Classifier for evaluation")

    if ByT5Classifier is None:
        raise ValueError("ByT5Classifier import failed.")

    id2label, label2id = load_label_maps(model_path)
    if label2id is None:
        raise ValueError("label2id not found in ByT5 model directory!")

    num_labels = len(label2id)

    tok = AutoTokenizer.from_pretrained("google/byt5-small")

    model = ByT5Classifier("google/byt5-small", num_labels)
    state = torch.load(f"{model_path}/pytorch_model.bin", map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    pred_ids, gold_ids = [], []

    for row in dataset:
        text = row["cs_query"]
        gold = row["domain"]

        tokens = tok(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            out = model(**tokens)

        pred_id = int(out["logits"].argmax(-1).item())
        gold_id = label2id.get(gold.lower())

        pred_ids.append(pred_id)
        gold_ids.append(gold_id)

    acc = accuracy_metric.compute(predictions=pred_ids, references=gold_ids)["accuracy"]
    f1s = f1_metric.compute(predictions=pred_ids, references=gold_ids, average="macro")["f1"]
    return acc, f1s


# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------
all_results = {}

for dataset_name, data_path in DATASETS.items():

    print("\n\n============================================================")
    print(f"🔄 Loading dataset: {data_path}")
    print("============================================================")

    ds = load_dataset("csv", data_files={"test": f"{data_path}/test.tsv"}, delimiter="\t")["test"]
    print(f"Loaded {len(ds)} samples.")

    sub_results = {}

    for model_name, model_path in MODELS.items():
        print(f"\n🔍 Evaluating {model_name.upper()} on {dataset_name.upper()}...")

        try:
            if model_name == "byt5":
                acc, f1s = evaluate_byt5(model_path, ds)
            else:
                acc, f1s = evaluate_mbert_or_canine(model_path, ds)
        except Exception:
            import traceback
            traceback.print_exc()
            acc, f1s = 0.0, 0.0

        sub_results[model_name] = {
            "accuracy": round(acc * 100, 2),
            "f1": round(f1s * 100, 2),
        }

    all_results[dataset_name] = sub_results


# ---------------------------------------------------------------------
# NRI CALCULATION (AGAINST CLEAN)
# ---------------------------------------------------------------------
final = {}
for model_name in MODELS.keys():
    clean = all_results["clean"][model_name]["accuracy"]

    final[model_name] = {
        "clean": clean,
    }

    for noise_tier in ["noisy_low", "noisy_medium", "noisy_high"]:
        noisy_acc = all_results.get(noise_tier, {}).get(model_name, {}).get("accuracy", 0)
        nri = (noisy_acc / clean * 100) if clean > 0 else 0
        final[model_name][noise_tier] = noisy_acc
        final[model_name][f"{noise_tier}_NRI"] = round(nri, 2)


Path(RESULTS_OUT).parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_OUT, "w") as f:
    json.dump(final, f, indent=2)

print("\n============================================================")
print("✅ Final NRI Calculation Complete!")
print(f"📁 Results saved to: {RESULTS_OUT}")
print("============================================================")
print(json.dumps(final, indent=2))
