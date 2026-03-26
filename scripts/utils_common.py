"""
utils_common.py
Shared utilities for Hinglish-NLP research pipeline.
Author: Yuvraj Arora
"""

import pandas as pd
import numpy as np
import difflib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------
# Canonical domain labels (sorted)
# ---------------------------------------------------------
DOMAIN_LABELS = ["alarm", "event", "messaging", "music", "navigation", "reminder", "timer", "weather"]

def fuzzy_match(pred: str, gold: str, threshold=0.8):
    """Returns True if pred ~ gold (>= threshold similarity)."""
    score = difflib.SequenceMatcher(None, pred, gold).ratio()
    return score >= threshold

def nearest_label(pred: str):
    """Maps any generated string to the closest canonical label."""
    pred = pred.strip().lower()
    best = None
    best_score = -1
    for label in DOMAIN_LABELS:
        score = difflib.SequenceMatcher(None, pred, label).ratio()
        if score > best_score:
            best_score = score
            best = label
    return best

def load_dataset_split(path):
    """Load a TSV dataset split."""
    return pd.read_csv(path, sep="\t")

def load_byt5_for_eval(model_path):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    return tok, model

def classify(model, tokenizer, text, max_new_tokens=20):
    """Generates label for given input text using ByT5."""
    inp = tokenizer(f"Classify the domain of: {text}",
                    return_tensors="pt",
                    truncation=True, max_length=256)

    out = model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=4)
    pred = tokenizer.decode(out[0], skip_special_tokens=True).strip().lower()
    return pred
