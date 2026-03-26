# Research Logbook — Token-Free Hinglish NLP  
Author: **Yuvraj Arora**

---

## 📘 Phase 0 — Initialization  
- Created folder structure  
- Generated integrity manifest  
- Locked environment  

---

## 📘 Phase 1 — Clean Training  
- Trained:
  - mBERT (95.7% accuracy)
  - CANINE (93.7%)
  - ByT5 (29.7%)  
- All training logs stored in `results/logs/`

---

## 📘 Phase 2 — Noise Robustness  
### Synthetic Noise (v2)
- mBERT crashed to 8.4%  
- CANINE dropped moderately  
- ByT5 retained ~23% accuracy  

### Human-Like Noise (Medium)
- mBERT collapsed (1.49%)
- CANINE strong stabilization (52.9%)
- **ByT5 almost invariant (27.3%)**

---

## 📘 Phase 3 — Final Analysis  
- Computed NRI  
- Generated plots  
- Wrote discussion section  

---

## 📘 Observations
- Token-based models fail on code-mixed noisy text  
- Byte-level modeling excels at multilingual noise  
- ByT5’s performance is consistent regardless of text corruption  

---

## 📘 Future Work
- Hybrid CANINE encoder + ByT5 decoder  
- Extend to Tamil-English, Marathi-English  
- RAG-enhanced factual grounding

