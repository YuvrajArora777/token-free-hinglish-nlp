# Token-Free Modeling for Hinglish NLP: A Noise-Robust Benchmark Using Byte and Character-Level Models

---

## 📄 IEEE Copyright Notice

**This work is copyrighted by IEEE and has been accepted for oral presentation at the 2026 IEEE International Conference on Interdisciplinary Approaches in Technology and Management for Social Innovation (IATMSI). The paper is currently under review for publication.**

**Copyright © 2026 IEEE. All rights reserved.**

### Authors' Retained Rights
Per IEEE policy, the authors (Yuvraj Arora, Vishwanath Bijalwan, Gurpreet Singh) retain the following rights:
- ✅ Post this accepted manuscript on institutional/personal servers with IEEE copyright notice
- ✅ Use portions for teaching, training, and future research
- ✅ Include material in derivative works and future publications
- ✅ Fulfill government funding deposit requirements

For full terms, see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

---

## Abstract

This repository studies robustness in Hinglish (Hindi-English code-mixed) text classification under realistic noise perturbations. We benchmark one token-based baseline (mBERT) against two token-free alternatives (CANINE-S and ByT5-small) and demonstrate that **token-free models significantly outperform subword tokenization under noisy inputs**. We introduce the **Noise Robustness Index (NRI)** as a standardized metric for evaluating model stability under realistic corruption (typos, casing shifts, emoji insertion, transliteration variation). Our findings highlight the importance of subword-free architectures for deploying reliable multilingual NLP systems in informal, noisy, low-resource contexts.

---

## 🎯 Motivation

### Problem Statement
- **Code-mixed Hinglish is ubiquitous** in real-world applications: social media, chat platforms, search queries, customer support.
- **Existing NLP systems degrade severely** under realistic noise (typos, switching case, mixed scripts, emoji use, slang abbreviations).
- **Token-based (subword) models are brittle**: they break when tokenization boundaries are corrupted by character-level noise.
- **Robustness under realistic noise is critical** for reliable deployment of multilingual NLP systems in informal language settings.

### Why This Matters
Current benchmarks (SQuAD, GLUE) focus on clean, formal text. Production systems face:
- Typos and spelling variations
- Mixed language code-switching
- Case inconsistency
- Emoji and special character injection
- Transliteration noise (Devanagari ↔ Latin)

Token-free models (character/byte-level) better handle these phenomena by preserving subword structure and morphological information.

---

## 📊 Key Insight

**Token-free models (CANINE-S, ByT5-small) significantly outperform subword-based models (mBERT) under noisy Hinglish inputs.**

- Performance gap **widens as noise intensity increases** (low → high).
- ByT5-small achieves **94.93% NRI in high-noise conditions**, vs. mBERT's **70.31%**.
- CANINE-S achieves **99.56% NRI in low-noise conditions**, demonstrating near-invariance to minor perturbations.

---

## 🔬 Contributions

1. **Introduced a Hinglish noise robustness benchmark** with realistic, humanlike perturbations (typos, emoji, casing, transliteration, phonetic corruptions, slang abbreviations).
2. **Proposed Noise Robustness Index (NRI)** as a standardized metric:
   ```
   NRI = (Noisy Accuracy / Clean Accuracy) × 100
   ```
   Enables direct comparison of model stability across architectures and noise levels.
3. **Performed systematic comparative evaluation** of:
   - Token-based (mBERT — subword tokenization)
   - Character-level (CANINE-S)
   - Byte-level (ByT5-small)
4. **Built a reproducible, modular evaluation pipeline** with step-by-step scripts, dataset validation, and integrity checking.

---

## 📈 Dataset Details

**Hinglish Sentiment Classification Corpus**

| Split | Samples | Perturbation Levels |
|-------|---------|-------------------|
| Train (clean) | 2,994 | — |
| Validation (clean) | 1,391 | — |
| Test (clean) | 6,514 | — |
| **Total Clean** | **10,899** | **1× (baseline)** |
| Test (noisy) | 6,514 | 3× (low, medium, high) |

**Noise Perturbation Strategy:**
Each test sample is corrupted at three severity levels using a combination of:

| Perturbation Type | Examples | Severity |
|------------------|----------|----------|
| **Phonetic Noise** | kya → kia, hai → he, nahi → nhi | All levels |
| **Keyboard Errors** | QWERTY neighbor mistakes (a→s, o→i) | All levels |
| **Typos** | Insertion, substitution, deletion | All levels |
| **Character Repetition** | hello → hellooo | Medium/High |
| **Case Variation** | RaNdOm MiXeD cAsE | All levels |
| **Emoji Injection** | "I love this" → "I 💯 love this" | Medium/High |
| **Slang Abbreviations** | you → u, please → plz, cuz → because | All levels |
| **Punctuation Noise** | ???, !!!, ?!? | Medium/High |
| **Whitespace Noise** | "hello world" → "hel lo world" | Low/Medium |

**Noise Intensity Levels:**
- **Low (12% corruption rate)**: 1-2 perturbations per sample
- **Medium (28% corruption rate)**: 2-4 perturbations per sample
- **High (50% corruption rate)**: 3+ perturbations per sample

---

## 📐 Noise Robustness Index (NRI) Definition

**NRI = (Accuracy_noisy / Accuracy_clean) × 100**

**Interpretation:**
- **NRI = 100%**: Perfect robustness (no performance degradation under noise).
- **NRI = 50%**: Model retains 50% of clean accuracy under noise.
- **NRI < 30%**: Poor robustness (significant performance collapse).

**Why NRI?** 
- Normalizes for model baseline performance (e.g., high-performing models can still have low NRI under attack).
- Enables fair comparison across different architectures and datasets.
- Reflects real-world durability: a model that degrades 10% is better than one that degrades 90%, regardless of absolute accuracy.

---

## 🧠 Why Token-Free Models Matter

| Aspect | Subword (mBERT) | Character (CANINE) | Byte-Level (ByT5) |
|--------|-----------------|------------------|------------------|
| **Tokenization** | Breaks on morphology | Preserves chars | Preserves bytes |
| **Typo resilience** | ❌ Fails | ✅ Strong | ✅ Very Strong |
| **Mixed scripts** | ❌ Weak | ✅ Good | ✅ Excellent |
| **Emoji handling** | ❌ OOV errors | ✅ Treated as char | ✅ Treated as bytes |
| **Morphology** | ✅ Good (for English) | ✅ Excellent | ✅ Excellent |
| **Multilingual** | ✅ Okay | ✅ Better | ✅ Best |

**Key Insight:** Subword tokenization is optimized for clean, formal, monolingual text. It breaks under character-level perturbations common in social media, chat, and low-resource settings. Character and byte-level models gracefully degrade.

---

## 🔧 Experimental Setup

### Models & Architecture
- **mBERT** (Multilingual BERT base): 12-layer transformer with subword WordPiece tokenization
- **CANINE-S** (Character-level Normalization-Insensitive Encoder): 12-layer transformer with character input preprocessing
- **ByT5** (Byte-level Text-to-Text Transfer Transformer, small variant): 6 encoder + 6 decoder layers, byte-level vocabulary

### Training Configuration

| Hyperparameter | mBERT | CANINE-S | ByT5-small |
|----------------|-------|----------|-----------|
| **Learning Rate** | 3e-5 | 3e-5 | 3e-4 |
| **Batch Size** | 8 | 2 + grad accum ×4 | 4 |
| **Epochs** | 4 | 4 | 4 |
| **Optimizer** | AdamW | AdamW | AdamW |
| **Warmup Steps** | 500 | 500 | 500 |
| **Max Seq Length** | 128 | 512 | 512 |

### Hardware & Frameworks
- **Framework:** PyTorch with Hugging Face Transformers
- **Hardware:** NVIDIA GPU (details in training logs)
- **Precision:** fp32 (stable training for byte-level models)

### Evaluation Protocol
1. Train on clean data only.
2. Evaluate on three test splits: clean, noisy (medium), noisy (high).
3. Compute accuracy per split.
4. Calculate NRI as ratio of noisy to clean accuracy.

---

## 📊 Evaluation Metrics

### Primary Metrics
- **Accuracy**: Classification accuracy on sentiment labels (binary: positive/negative).
- **Noise Robustness Index (NRI)**: Relative performance retention under corruption.
- **F1-Score**: Harmonic mean of precision and recall (reported where applicable).

### Secondary Metrics
- **Robustness Curve**: How accuracy degrades across noise levels (low → medium → high).
- **Per-Class NRI**: Are some sentiment classes more robust than others?

---

## 📈 Key Results

### High-Noise Condition (Strictest Robustness Test)

Performance under 50% corruption rate:

| Model | Clean Acc | Noisy High Acc | NRI High (%) |
|-------|-----------|----------------|----|
| **mBERT** | 94.86% | 66.70% | **70.31%** |
| **CANINE-S** | 93.41% | 87.92% | **94.12%** |
| **ByT5-small** | 94.89% | 90.08% | **94.93%** |

**Interpretation:**
- mBERT degrades from 94.86% → 66.70% (−28.16 pp loss).
- CANINE-S retains 87.92% (only −5.49 pp loss).
- **ByT5-small achieves highest NRI (94.93%), losing only 4.81 pp from clean baseline.**

### Medium & Low Noise Levels

| Model | Noisy Low Acc | NRI Low (%) | Noisy Medium Acc | NRI Medium (%) |
|-------|----------------|-------------|------------------|----------------|
| **mBERT** | 87.78% | 92.54% | 79.49% | 83.80% |
| **CANINE-S** | 92.48% | 99.00% | 91.20% | 97.63% |
| **ByT5-small** | 94.47% | 99.56% | 92.68% | 97.67% |

### Robustness Ranking
1. **ByT5-small** (high NRI even under severe noise)
2. **CANINE-S** (excellent low/medium noise resilience, strong high-noise retention)
3. **mBERT** (degrades significantly; best for clean data only)

### Metrics Source
- File: [results/metrics/final_nri.json](results/metrics/final_nri.json)
- Detailed: [results/metrics/](results/metrics/)

---

## 🔍 Analysis: Why Token-Free Models Win

1. **Character/Byte Granularity**: Absorb character-level corruption without breaking tokenization boundaries.
2. **Morphological Transparency**: Preserve structure of Hinglish code-switching and informal spelling.
3. **Graceful Degradation**: Performance drops gradually, not catastrophically.
4. **No OOV (Out-of-Vocabulary) Problem**: Every possible byte/character is in vocabulary.

**mBERT's Weakness:**
- WordPiece tokenization depends on exact character sequences.
- Typos → different subword tokens → out-of-distribution for training.
- Code-mixing breaks tokenization (Hindi script + English script adjacent).

---

## 🧪 Reproducibility Setup

### Prerequisites
- Python 3.10+
- GPU with ≥12GB VRAM (recommended) or CPU (slow)
- ~50GB disk space for models and artifacts

### Installation
```bash
# Clone repository
git clone https://github.com/YuvrajArora777/token-free-hinglish-nlp.git
cd token-free-hinglish-nlp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Full Pipeline Execution
Run scripts in order (each step depends on previous output):

```bash
# 0️⃣ Reset project state
python scripts/00_reset_project.py

# 1️⃣ Validate and prepare clean dataset
python scripts/01_create_datasets.py

# 2️⃣ Generate noisy variants (low, medium, high)
python scripts/02_generate_noise.py

# 3️⃣ Train mBERT on clean data
python scripts/03_train_mbert.py

# 4️⃣ Train CANINE-S on clean data
python scripts/04_train_canine.py

# 5️⃣ Train ByT5-small on clean data
python scripts/05_train_byt5.py

# 6️⃣ Evaluate all models on clean + noisy sets
python scripts/06_evaluate_models.py

# 7️⃣ Generate robustness charts and visualizations
python scripts/07_visualize_results.py

# 8️⃣ Evaluate with predictions (additional metrics)
python scripts/08_evaluate_with_predictions.py

# 9️⃣ Generate comprehensive visualizations
python scripts/09_visualize_all.py
```

### Reproducibility Guarantees
- **Deterministic Pipeline**: All random seeds are fixed (`scripts/utils_common.py`).
- **Dataset Validation**: SHA-256 integrity manifest (`integrity_manifest.json`).
- **Environment Lock**: Pin all dependencies (`configs/env_lock.txt`).
- **Modular Design**: Each script is self-contained and can be re-run independently.

---

## 📁 Repository Structure

```
hinglish-nlp/
├── configs/
│   └── env_lock.txt              # Pinned dependency versions
├── data/                         # ⚠️ NOT in GitHub (regenerate via scripts)
│   ├── human_annotated/
│   │   ├── train.tsv
│   │   ├── validation.tsv
│   │   └── test.tsv
│   ├── noisy_humanlike_low/
│   ├── noisy_humanlike_medium/
│   └── noisy_humanlike_high/
├── models/                       # ⚠️ NOT in GitHub (regenerate via scripts)
│   ├── mbert/
│   ├── canine/
│   └── byt5/
├── results/                      # ⚠️ NOT in GitHub (regenerate via scripts)
│   ├── metrics/
│   │   ├── final_nri.json
│   │   ├── mbert_metrics.json
│   │   ├── canine_metrics.json
│   │   └── byt5_metrics.json
│   ├── charts/
│   ├── logs/
│   └── predictions/
├── scripts/
│   ├── 00_reset_project.py
│   ├── 01_create_datasets.py
│   ├── 02_generate_noise.py
│   ├── 03_train_mbert.py
│   ├── 04_train_canine.py
│   ├── 05_train_byt5.py
│   ├── 06_evaluate_models.py
│   ├── 07_visualize_results.py
│   ├── 08_evaluate_with_predictions.py
│   ├── 09_visualize_all.py
│   ├── utils_common.py
│   └── byt5_model.py
├── integrity_manifest.json
├── project_report.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

---

## 💾 Artifact Availability

To keep the public GitHub repository lightweight and CI/CD-friendly, large artifacts are **not** versioned:
- `data/` (~100 MB per split × 3 noise levels)
- `models/` (~5+ GB per model × 3 models)
- `results/` (logs, predictions, visualizations)

### Reproducible Regeneration
All artifacts are **deterministically regenerated** by running the scripts in order:
```bash
python scripts/00_reset_project.py  # Clears all artifacts
python scripts/01_create_datasets.py ... python scripts/09_visualize_all.py
```

### For Camera-Ready Submission
Upon paper acceptance, artifacts will be archived to:
- **Zenodo** (for long-term preservation with DOI)
- **Hugging Face Datasets Hub** (for easy access and versioning)
- Link will be updated in README and CITATION.cff

---

## 🛡 Reproducibility Statement

This project prioritizes **reproducibility and transparency**:

✅ **Fully reproducible pipeline** with modular, step-by-step scripts.  
✅ **Deterministic setup**: Fixed random seeds across all experiments.  
✅ **Environment lock file**: Pinned dependency versions for exact recreation.  
✅ **Dataset validation**: SHA-256 checksums for input/output integrity.  
✅ **Modular design**: Each script is independently executable.  
✅ **Complete hyperparameter transparency**: All training configs in code.  
✅ **Training logs**: Saved per-epoch metrics in `results/logs/`.  

---

## ⚠️ Limitations

1. **Dataset Size**: 10,899 clean samples is modest compared to billion-scale pretraining corpora. Results may not directly transfer to much larger Hinglish corpora.
2. **Synthetic Noise**: Realistic noise generation, but may not capture all real-world variations (e.g., non-Latin transliteration, domain-specific slang, cross-platform encoding issues).
3. **Model Size**: Only small variants tested (mBERT-base, CANINE-S, ByT5-small) due to GPU constraints. Larger models (e.g., XLM-R, CANINE-large, ByT5-base) may show different patterns.
4. **Binary Classification**: Sentiment classification only. Results may differ for other tasks (NER, parsing, question-answering).
5. **Language Pair**: Hinglish only. Generalization to other code-mixed pairs (Tanglish, Spanglish, Franais, etc.) is unexplored.
6. **Training Data**: All models trained exclusively on **clean** data. Robustness-aware training strategies (adversarial training, mixup) are not explored.

---

## 🚀 Future Work

1. **Extend to other code-mixed pairs**: Tanglish (Tamil-English), Benglish (Bengali-English), apply same pipeline.
2. **Larger-scale datasets**: Collect or crowdsource larger Hinglish corpora.
3. **Robustness-aware training**: Adversarial training, noise injection during training, data augmentation.
4. **Newer architectures**: Evaluate larger models (XLM-R, mT5-base, CANINE-large), instruction-tuned LLMs (LLaMA, Mistral finetuned on Hinglish).
5. **Domain adaptation**: Test on social media, customer support, domain-specific text.
6. **Hybrid architectures**: CANINE encoder + ByT5 decoder for combining character-level input with byte-level generation.
7. **Explainability**: Attention analysis to understand why token-free models are more robust.

---

## 📝 Citation

If you use this repository, please cite the work as:

```bibtex
@software{arora2026hinglish,
  title={Token-Free Modeling for Hinglish NLP: A Noise-Robust Benchmark Using Byte and Character-Level Models},
  author={Arora, Yuvraj},
  year={2026},
  url={https://github.com/YuvrajArora777/token-free-hinglish-nlp},
  version={1.0.0}
}
```

Or use the CITATION.cff file directly.

---

## 📄 Paper Status

**Paper:** Under submission (IEEE) — will be updated upon acceptance.

Contributions welcome via:
- Issues: Bug reports, questions, clarifications
- Pull Requests: Code improvements, additional experiments

---

## 📧 Contact

For questions, suggestions, or collaboration:
- Author: **Yuvraj Arora**
- GitHub: [@YuvrajArora777](https://github.com/YuvrajArora777)

---

## License

MIT License — See [LICENSE](LICENSE) for full text.

