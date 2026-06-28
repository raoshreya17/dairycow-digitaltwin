# dairycow-digitaltwin — MooAnalytica

AI-driven digital twin of dairy cows integrating behaviour perception, nutrition modelling, and closed-loop feeding management. Developed at Dalhousie University (Ruminant Animal Centre, Truro NS, March 2025).
---

## Overview

The system processes continuous barn CCTV video through a four-stage pipeline:
1. **Detection** — YOLOv11 detects individual Holstein cows in each frame (mAP@50 = 0.994)
2. **Tracking** — ByteTrack with stall zone anchoring maintains per-cow identity across sessions
3. **Classification** — TimeSformer classifies seven behaviour classes (accuracy 85.8%, macro-F1 0.836, 22.6 fps)
4. **Nutrition + Control** — Michaelis-Menten DMI estimation with proportional feedback controller for per-cow feed allocation

---

## Repository Structure
```
dairycow-digitaltwin/
├── Detection/              YOLOv11 training and inference (2308 images, Holstein RAC dataset)
├── Tracking/               ByteTrack with stall zone anchoring for identity persistence
│   ├── ByteTrack.ipynb
│   ├── ExtractCowsCrops.ipynb
│   └── extractclipfromfolder.ipynb
├── Classification/         TimeSformer and SlowFast training notebooks (outputs cleared; weights on HuggingFace)
│   ├── timeSformer_training.ipynb
│   └── SlowFast_training.ipynb
├── Explainability/         GradCAM attention map visualisation
│   ├── FinalGradCamSF.ipynb
│   └── explainabilityforallclasses.py
├── Evaluation/             Robustness and ablation evaluation
│   ├── RobustnessDayNight.ipynb   Day/night stratified evaluation (R-B threshold=2.0)
│   └── TemporalAblation.ipynb     12 vs 16 vs 32 frame temporal sampling ablation
├── Pipeline/               End-to-end inference: detection → tracking → classification → CSV
│   ├── pipeline.py
│   └── Complete_pipeline_with_TimeSformer.ipynb
├── Nutrition/              DMI estimation models, closed-loop controller, simulation (Chapter 6)
│   ├── NutritionModel.ipynb             FCM-corrected M-M model (Km=30, MAPE=3.2%, R²=0.770)
│   └── controller_simulation_final.ipynb  500-run controller stability analysis
├── TimeSformer 2 head/     Two-head TimeSformer variant (experimental)
├── Models/                 Model weights and configs
│   ├── best.pt             YOLOv11 weights (19 MB)
│   ├── bytetrack.yaml      ByteTrack configuration
│   └── readme.md           TimeSformer HuggingFace link
├── requirements.txt
└── .gitignore
```

---

## Pipeline
```
CCTV video (1920×1080, 24 fps)
    │
YOLOv11 detection  (best.pt)
    │  bounding boxes + confidence
ByteTrack + stall zone anchoring
    │  cow_id, tracklet, per-frame bbox
TimeSformer classification  (10s clips, 16 frames, 224×224)
    │  {cow_id, behaviour, start, end, duration, confidence}
CSV behavioural timelines
    │  daily feeding_min, rumination_min, lying_min per cow
DMI estimation
    │  DMI_est(t), DMI_NRC(t), e(t)
Proportional feedback controller
    │  u(t+1) = clip(u(t) + K·e(t), 15, 27.5)  kg DM/day
Feeding recommendations → Unity 3D Digital Twin HMI
```

---

## Behaviour Classes

Seven classes recognised by TimeSformer and SlowFast:

| ID | Class | Description |
|----|-------|-------------|
| 0 | Drinking | At water trough |
| 1 | Feeding & Lying | Active feeding, lying posture |
| 2 | Feeding & Standing | Active feeding at the bunk, standing |
| 3 | Lying | Resting, no feeding or rumination |
| 4 | Ruminating & Lying | Jaw movement, lying |
| 5 | Ruminating & Standing | Jaw movement, standing |
| 6 | Standing | Stationary, not feeding |

---

## Performance

| Component | Metric | Value |
|-----------|--------|-------|
| YOLOv11 detection | mAP@50 | 0.994 |
| TimeSformer classification | Overall accuracy | **85.8%** |
| TimeSformer classification | Macro-F1 | **0.836** |
| TimeSformer classification | Daytime accuracy (n=670) | 88.2% |
| TimeSformer classification | Nighttime accuracy (n=449) | 81.5% |
| TimeSformer classification | mAP (macro) | 89.0% |
| SlowFast classification | Overall accuracy | 85.0% |
| SlowFast classification | Macro-F1 | 0.821 |
| Combined pipeline | Throughput | 77.4 fps (daytime) |
| Combined pipeline | Latency | 180 ms per frame |
| M-M DMI model (whole-dataset) | MAPE | **3.2%** |
| M-M DMI model (whole-dataset) | R² | **0.770** |
| M-M DMI model (Km) | Half-saturation constant | **30 min/day** |
| Regression DMI model (LOCO, 16 cows) | MAPE | 5.8% |
| Regression DMI model (LOCO, 16 cows) | R² | 0.313 |
| Controller (500 runs, 17% model-plant mismatch) | Convergence rate | **100%** |
| Controller | Median recovery time | **3.0 days** |
| Controller | 95th-percentile recovery time | **7.0 days** |

---

## Nutrition Module

The `Nutrition/` folder contains the full Chapter 6 analysis across 16 cows, 153 cow-days (March 2025, RAC Truro NS). All models use **Fat-Corrected Milk (FCM)** per NRC (2001).

Three models evaluated:

- **Linear baseline** — `DMI = k × feeding_min` (MAPE 15.0% local refit; 31.8% Johnston & DeVries 2018 transfer)
- **Pooled regression (LOCO)** — `DMI = a + b×feeding_min + c×BW + d×FCM`, LOCO MAPE 5.8%, R²=0.313
- **Michaelis-Menten** — `DMI = DMI_MAX × f / (Km + f)`, Km=30 min/day, MAPE=3.2%, R²=0.770

Closed-loop proportional feedback controller (model-plant mismatch: plant Km=35, estimator Km=30):

```
u(t+1) = clip( u(t) + K·e(t)·[|e(t)| > δ],  u_min,  u_max )
K = 0.5,  δ = 0.5 kg DM/day,  u ∈ [15, 27.5] kg DM/day
```

500-run simulation (100 seeds × 5 lactation scenarios): 100% convergence within 14 days, median recovery 3.0 days.

---

## Model Weights

| Model | Location |
|-------|----------|
| YOLOv11 (best.pt, 19 MB) | `Models/best.pt` |
| TimeSformer fine-tuned (last.pt, 1.46 GB) | [huggingface.co/shreyayayay/timesformer-dairy-cows](https://huggingface.co/shreyayayay/timesformer-dairy-cows) |

Download `last.pt` and set `MODEL_DIR` in `Pipeline/pipeline.py` accordingly.

---

## Data

RAC barn video is not publicly released. The annotated clip dataset (4,964 clips, train/val/train_aug) and processed CSVs are available via a shared Dropbox folder.

---

## Hardware

| Stage | Hardware |
|-------|----------|
| TimeSformer training | NVIDIA A100 40GB (CUDA 12.4, PyTorch 2.0) |
| Pipeline inference | NVIDIA RTX PRO 4000 (CUDA 11.6, PyTorch 1.12.0) |

---

## Dependencies

See `requirements.txt`. Key packages:

```
python >= 3.10
torch >= 2.0
ultralytics          # YOLOv11
transformers==4.41.2 # TimeSformer
supervision          # ByteTrack
decord==0.6.0        # video decoding
numpy, pandas, scipy, matplotlib, scikit-learn
xlrd                 # .xls milk yield files
```

---

## Citation

```bibtex
@article{rao2026mooanalytica,
  title   = {An AI-Driven Behavioural Perception and Nutrition Modelling Framework
             as the Behavioural State Layer of a Dairy Cattle Digital Twin},
  author  = {Rao, Shreya and Garcia, Carlos and Neethirajan, Suresh},
  journal = {npj Veterinary Sciences},
  volume  = {1},
  number  = {3},
  year    = {2026},
  doi     = {10.1038/s44433-026-00004-x}
}
```

## License

MIT — see [LICENSE](LICENSE)
    v
Feeding recommendations → Unity 3D HMI
```

## Behaviour Classes

Seven classes recognised by TimeSformer:

| Class | Description |
| Feeding & Standing | Active feeding at the bunk, standing posture |
| Feeding & Lying | Active feeding, lying posture |
| Drinking | At water trough |
| Lying | Resting, no feeding |
| Standing | Stationary, not feeding |
| Ruminating & Standing | Jaw movement, standing |
| Ruminating & Lying | Jaw movement, lying |

## Performance

| Component | Metric | Value |
| YOLOv11 detection | mAP@50 | 0.994 |
| TimeSformer classification | Overall accuracy | 85.0% |
| TimeSformer classification | Macro-F1 | 0.841 |
| Combined pipeline | Throughput | 77.4 fps (daytime) |
| Combined pipeline | Latency | 180 ms per frame |
| M-M DMI model (5 cows, 55 days) | RMSE | 0.606 kg DM/day |
| M-M DMI model (5 cows, 55 days) | MAPE | 1.5% |
| Regression DMI model (16 cows, LOCO) | MAPE | 5.8% |
| Controller convergence | Recovery time | 2–8 days |

## Nutrition Module

The `Nutrition/` folder contains Chapter 6 of the thesis in full.

Three models are evaluated against NRC (2001) as physiological reference across 16 cows, 153 cow-days (March 2025, RAC facility, Truro NS):

- **Linear baseline** — `DMI = k * feeding_min`, k refit locally (MAPE 15.0%) and from Johnston & DeVries 2018 (MAPE 31.8%)
- **Pooled regression** — `DMI = a + b*feeding_min + c*BW + d*ECM`, primary model, LOCO MAPE 5.8%
- **Michaelis-Menten** — `DMI = DMI_MAX * f / (Km + f)`, per-cow ceiling, Km=40 min/day, MAPE 1.3% (whole-dataset)

Closed-loop proportional feedback controller:

```
u(t+1) = clip( u(t) + K * e(t) * [|e(t)| > delta],  u_min,  u_max )
K = 0.5,  delta = 0.5 kg DM/day,  u in [15, 28] kg DM/day
```

## Data

The RAC barn video dataset is not publicly available due to institutional data agreements. The pipeline CSV outputs (behavioural timelines, DMI estimates, model results) used in the paper are available on request.

Cow physiological parameters (BW, DIM, fat%, protein%, parity) for the 17-cow study cohort are hardcoded in `Nutrition/chapter6_mm_complete.py`. Milk yield records are read from the milking system Excel export (`Copy of milk yield.xls`).

## Hardware

All video inference was run on a cloud GPU instance (NVIDIA RTX PRO 4000, 23.9 GB VRAM, CUDA 13.0, AMD EPYC 7B13 CPU, 64.4 GB RAM) provisioned through Vast AI. Total compute cost for the 11-day, 5-cow dataset was approximately $32 USD.

## Dependencies

```
python >= 3.10
torch >= 2.0
ultralytics          # YOLOv11
transformers         # TimeSformer, HuggingFace
supervision          # ByteTrack
numpy
pandas
scipy
matplotlib
scikit-learn
xlrd                 # reading .xls milk yield files
```
## Model Weights

TimeSformer fine-tuned checkpoint (1.4 GB):
https://huggingface.co/shreyayayay/timesformer-dairy-cows

Download and place at the path set in MODEL_DIR in pipeline.py.
