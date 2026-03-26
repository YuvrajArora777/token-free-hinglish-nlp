"""
ByT5 Domain Classification - OPTIMIZED FOR 80%+ ACCURACY
Based on transcript understanding: byte-level encoder + proper tuning

Key insights:
- ByT5 byte-level encoding preserves Hinglish nuances
- Use only encoder (classification, not generation)
- Higher LR (3e-4) needed for byte-level sparse gradients
- fp32 training for numerical stability
- Pool first token [CLS] equivalent for classification
- 4 epochs with proper learning rate scheduling
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
    Trainer,
    TrainingArguments,
)

# --- Constants ---
MODEL_NAME = "google/byt5-small"
MODEL_ALIAS = "byt5"
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


# --- ByT5 Classifier: Encoder-only with byte-level tokenization ---
class ByT5Classifier(nn.Module):
    """
    Byte-level T5 encoder for classification.
    
    ByT5 operates directly on UTF-8 bytes (0-255), preserving:
    - Typos and misspellings
    - Mixed-script Hinglish
    - Emojis and special characters
    
    Architecture:
    - T5EncoderModel: byte-level transformer encoder
    - First token pooling (CLS equivalent)
    - Linear classification head
    - No decoder (classification task, not generation)
    """

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.d_model, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for classification.
        
        Args:
            input_ids: Byte-level token IDs (0-255 + special tokens)
            attention_mask: Mask for padding
            labels: True class labels for loss computation
            
        Returns:
            Dictionary with "loss" and "logits"
        """
        # Encode byte sequences
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Pool: use first token [CLS] equivalent
        # This aggregates the byte-sequence representation
        pooled_output = encoder_output.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Classification logits
        logits = self.classifier(pooled_output)
        
        # Loss computation
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 8), labels.view(-1).long())
        
        return {"loss": loss, "logits": logits}


def main():
    """Main training pipeline."""
    
    # --- Create Directories ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load Dataset ---
    print("📂 Loading Hinglish-TOP dataset...")
    data_files = {
        "train": str(DATA_DIR / "train.tsv"),
        "validation": str(DATA_DIR / "validation.tsv"),
        "test": str(DATA_DIR / "test.tsv"),
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    print(f"✓ Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")

    # --- Prepare Labels ---
    labels = sorted(list(set(dataset["train"][LABEL_COLUMN])))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(labels)
    
    print(f"\n🏷️  Labels: {labels}")
    print(f"Num labels: {num_labels}")

    # --- Tokenizer & Preprocessing ---
    print(f"\n💾 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        """
        Tokenize Hinglish text at byte level.
        
        ByT5 tokenizer:
        - No vocabulary (token-free)
        - Direct byte-to-integer conversion
        - Robust to typos, mixed scripts, emojis
        """
        tokens = tokenizer(
            examples[TEXT_COLUMN],
            padding="max_length",
            truncation=True,
            max_length=256  # Byte sequences can be longer
        )
        tokens["labels"] = [label2id[label] for label in examples[LABEL_COLUMN]]
        return tokens

    print(f"🔄 Tokenizing dataset (byte-level)...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    print("✓ Tokenization complete")

    # --- Model ---
    print(f"\n🤖 Loading ByT5 encoder-only model: {MODEL_NAME}")
    model = ByT5Classifier(MODEL_NAME, num_labels)
    print(f"✓ Model loaded: {type(model).__name__}")

    # --- Metrics ---
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        """Compute accuracy and macro F1."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        
        return {"accuracy": acc, "f1": f1}

    # --- Training Arguments (Optimized for ByT5) ---
    # Key insight: ByT5 needs higher LR due to byte-level sparse gradients
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,               # HIGHER LR for byte-level models
        per_device_train_batch_size=4,    # GPU limitation
        per_device_eval_batch_size=8,
        num_train_epochs=4,               # 4 epochs for convergence
        weight_decay=0.01,                # L2 regularization
        fp16=False,                       # fp32 for numerical stability
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        seed=SEED,
        max_grad_norm=1.0,                # Gradient clipping
        warmup_ratio=0.1,                 # Warmup for stable training
        save_safetensors=False,           # T5Encoder has shared weights
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # --- Training ---
    print("\n" + "="*70)
    print("🚀 STARTING BYT5 FINE-TUNING (BYTE-LEVEL ENCODER-ONLY)")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Architecture: Byte-level T5 Encoder + Linear Classifier")
    print(f"Batch size: 4 | LR: 3e-4 | Epochs: 4 | fp32 (stable)")
    print(f"Input: cs_query (Hinglish code-mixed)")
    print(f"Task: Domain classification (8 classes)")
    print("Expected accuracy: ~80-82% on clean data")
    print("="*70 + "\n")

    trainer.train()

    # --- Evaluation ---
    print("\n" + "="*70)
    print("✅ FINAL EVALUATION ON TEST SET")
    print("="*70)

    results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test Results:")
    print(f"  Accuracy: {results.get('eval_accuracy', 0):.4f}")
    print(f"  F1 (macro): {results.get('eval_f1', 0):.4f}")

    # --- Save Metrics ---
    metrics_path = METRICS_DIR / "byt5_metrics.json"
    metrics_to_save = {
        "test_accuracy": float(results.get("eval_accuracy", 0)),
        "test_f1_macro": float(results.get("eval_f1", 0)),
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_path}")

    # --- Save Model ---
    final_model_path = OUTPUT_DIR / "final"
    final_model_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_path))

    # Save config with label mappings and set decoder_start_token_id to None
    # This is crucial for loading with AutoModelForSequenceClassification
    config = model.encoder.config
    config.label2id = label2id
    config.id2label = id2label
    config.decoder_start_token_id = None  # Mark as encoder-only for classification
    config.save_pretrained(str(final_model_path))

    print(f"✓ Model saved to {final_model_path}")
    
    print("\n" + "="*70)
    print("✓ ByT5 Training Pipeline Complete!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
