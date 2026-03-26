# Token-Free Modeling for Hinglish NLP

## Abstract
This repository studies robustness in Hinglish (Hindi-English code-mixed) text classification under realistic noise. We compare one token-based baseline (mBERT) with token-free alternatives (CANINE-S and ByT5-small) across clean and noisy test conditions. The core objective is to quantify robustness degradation due to typos, casing shifts, emoji insertion, and transliteration variation.

## Research Question
How much robustness do token-free models retain under noisy Hinglish compared with subword tokenization baselines?

## Model Family Comparison
- mBERT: subword tokenization baseline
- CANINE-S: character-level token-free model
- ByT5-small: byte-level token-free model

## Key Results
Final Noise Robustness Index (NRI) from `results/metrics/final_nri.json`.

High-noise condition (strictest split):

| Model | Clean Acc | Noisy High Acc | NRI High (%) |
|------|-----------|----------------|--------------|
| mBERT | 94.86% | 66.70% | **70.31%** |
| CANINE-S | 93.41% | 87.92% | **94.12%** |
| ByT5-small | 94.89% | 90.08% | **94.93%** |

All noise levels:

| Model | Noisy Low Acc | NRI Low (%) | Noisy Medium Acc | NRI Medium (%) |
|------|----------------|-------------|------------------|----------------|
| mBERT | 87.78% | 92.54% | 79.49% | 83.80% |
| CANINE-S | 92.48% | 99.00% | 91.20% | 97.63% |
| ByT5-small | 94.47% | 99.56% | 92.68% | 97.67% |

Interpretation:
- All models degrade with increasing noise, but token-free models remain more stable.
- CANINE-S and ByT5-small outperform mBERT under medium and high perturbation settings.
- ByT5-small has the highest high-noise NRI in this run.

## Method Summary
1. Build validated clean train/validation/test splits from human-annotated data.
2. Generate multiple noisy variants with humanlike perturbations.
3. Train three architectures independently.
4. Evaluate each model on clean and noisy partitions.
5. Compute model-wise metrics and NRI.
6. Generate visual comparisons and robustness charts.

## Reproducibility Setup
Use Python 3.10+ and install dependencies:

```bash
pip install -r requirements.txt
```

Run the end-to-end pipeline in order:

```bash
python scripts/00_reset_project.py
python scripts/01_create_datasets.py
python scripts/02_generate_noise.py
python scripts/03_train_mbert.py
python scripts/04_train_canine.py
python scripts/05_train_byt5.py
python scripts/06_evaluate_models.py
python scripts/07_visualize_results.py
python scripts/08_evaluate_with_predictions.py
python scripts/09_visualize_all.py
```

## Repository Structure
```text
hinglish-nlp/
|- configs/
|- data/                        # ignored in GitHub snapshot
|- models/                      # ignored in GitHub snapshot
|- results/                     # ignored in GitHub snapshot
|- scripts/
|- integrity_manifest.json
|- project_report.md
|- requirements.txt
```

## Artifact Availability
To keep the repository lightweight and GitHub-compatible, large artifacts are not versioned in this public snapshot:
- data/
- models/
- results/

These can be shared separately (for example, cloud storage, Hugging Face, or Zenodo) in a camera-ready update.

## Paper Status
Paper: Under submission (IEEE) and will be updated upon acceptance.

## Citation
If you use this repository, please cite the associated paper metadata from CITATION.cff.

