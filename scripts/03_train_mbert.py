# scripts/03_train_mbert.py
"""
Fine-tunes a multilingual BERT model for Hinglish text classification.

This script handles the entire workflow:
1. Loads the human-annotated dataset.
2. Prepares labels and tokenizes the text.
3. Sets up the BERT model for sequence classification.
4. Defines training arguments and a metrics computation function.
5. Trains the model using the Trainer API.
6. Evaluates the best model on the test set.
7. Saves the final model and evaluation metrics.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# --- Constants ---
MODEL_NAME = "bert-base-multilingual-cased"
MODEL_ALIAS = "mbert"
OUTPUT_DIR = Path(f"models/{MODEL_ALIAS}")
METRICS_DIR = Path("results/metrics")
DATA_DIR = Path("data/human_annotated")
TEXT_COLUMN = "cs_query"
LABEL_COLUMN = "domain"

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def main():
    """Main function to run the training and evaluation pipeline."""
    # --- Create Directories ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load Dataset ---
    print("Loading datasets...")
    data_files = {
        "train": str(DATA_DIR / "train.tsv"),
        "validation": str(DATA_DIR / "validation.tsv"),
        "test": str(DATA_DIR / "test.tsv"),
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    print(f"Dataset loaded: {dataset}")

    # --- Prepare Labels ---
    labels = sorted(list(set(dataset["train"][LABEL_COLUMN])))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(labels)
    print(f"Labels prepared: {num_labels} labels -> {label2id}")

    # --- Tokenizer and Preprocessing ---
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        """Tokenize text and map labels to IDs."""
        tokens = tokenizer(examples[TEXT_COLUMN], padding="max_length", truncation=True)
        tokens["labels"] = [label2id[label] for label in examples[LABEL_COLUMN]]
        return tokens

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # --- Model ---
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # --- Metrics ---
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        """Compute accuracy and F1 score."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        return {"accuracy": acc, "f1": f1}

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        num_train_epochs=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=["none"],
        fp16=torch.cuda.is_available(), # Enable mixed-precision if CUDA is available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # --- Evaluation ---
    print("Evaluating on the test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test results: {test_results}")

    # --- Save Artifacts ---
    metrics_path = METRICS_DIR / "mbert_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    final_model_path = OUTPUT_DIR / "final"
    trainer.save_model(str(final_model_path))
    print(f"Final model saved to {final_model_path}")
    print("Done.")

if __name__ == "__main__":
    main()
