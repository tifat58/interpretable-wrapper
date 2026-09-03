# Improvement Plan

Branch model: `stable` = frozen demo snapshot · `dev` = working branch · merge to `main` when validated.

## Phase 1 — Medical use case: from demo to paper-grade (done)

- [x] **1. Real/weak concept labels** (replace Bernoulli-sampled labels in `scripts/train_probes_real.py`)
  - [x] Pathology-aligned concepts (Consolidation, GGO, Lung Opacity, Effusion, Cardiomegaly, Edema, Atelectasis): weak labels from an *independent* torchxrayvision checkpoint (`densenet121-res224-all`) — different weights than the wrapped `-chex` model to reduce circularity
  - [x] Geometric concepts (Bilateral Involvement, Peripheral Distribution, Lung Volume Loss, Clear Lung Fields): derived from lung segmentation masks + intensity statistics (COVID-QU-Ex masks), z-scored against the Normal-class training distribution
  - [x] Air Bronchogram: conjunction proxy (consolidation ∧ pneumonia) — documented as weakest label
  - [x] New script: `backend/scripts/train_medical_cbm.py` with feature caching
- [x] **2. Trained task head** (replace hardcoded 0.4/0.3/0.3 heuristic in `models/medical_model.py`)
  - [x] Logistic head on 1024-d DenseNet features → COVID-19 / Non-COVID / Normal, trained on COVID-QU-Ex Train split
  - [x] Temperature scaling on Val split; save to `backend/data/cbm/medical/task_head.pkl`
  - [x] `MedicalModel.predict_raw` loads the head when available, falls back to heuristic otherwise
  - [x] Standardize labels to `COVID-19 / Non-COVID / Normal` (config.py, counterfactuals.py)
- [x] **3. Proper evaluation** (currently train == test)
  - [x] Use the dataset's official Train/Val/Test splits everywhere
  - [x] Task head: accuracy, per-class AUC, macro-AUC, ECE before/after calibration
  - [x] Probes: per-concept val/test AUC + calibration (temperature folded into LR weights)
  - [x] Surrogate: fidelity (agreement with black-box head) on Test + ground-truth accuracy
  - [x] Persist all metrics to `backend/data/cbm/medical/metrics.json`

Results (6000 train / 1998 val / 1998 test): head test acc 0.811, macro-AUC 0.936, ECE 0.021;
probe test AUC 0.88–0.98 (all 12 trained); surrogate fidelity 0.75 (improvement target for Phase 2).
Re-run: `cd backend && python -m scripts.train_medical_cbm [--full]`

## Phase 2 — Explanation quality (next)

- [x] Probe-gradient GradCAM: concept probe coefficients weight DenseNet feature maps directly instead of using the hardcoded concept→pathology map (`medical_model.py`); verified live with a 224×224 heatmap
- [ ] Counterfactuals from surrogate coefficients instead of guessed weights (`counterfactuals.py`)
- [ ] Expand medical concept bank: Pneumothorax, Nodule, Mass, Fibrosis (already available as txv outputs)
- [ ] Show surrogate fidelity + per-concept probe reliability (val AUC) in the UI
- [ ] Per-concept uncertainty display (flag low-AUC probes as unreliable)

## Phase 3 — Engineering

- [ ] Smoke tests for `/predict`, `/attribution`, `/counterfactual` per domain; CI on `dev`
- [ ] Optional upgrade path: CheXpert/NIH real per-pathology labels for probe training (large download, replaces weak labels)
- [ ] Apply the same eval/calibration treatment to birds and vision domains
