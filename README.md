# dairycow-digitaltwin - MooAnalytica

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

- Linear baseline — `DMI = k × feeding_min` (MAPE 15.0% local refit; 31.8% Johnston & DeVries 2018 transfer)
- Pooled regression (LOCO) — `DMI = a + b×feeding_min + c×BW + d×FCM`, LOCO MAPE 5.8%, R²=0.313
- Michaelis-Menten — `DMI = DMI_MAX × f / (Km + f)`, Km=30 min/day, MAPE=3.2%, R²=0.770

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

RAC barn video is not publicly released due to institutional data agreements. The annotated clip dataset (4,964 clips, train/val/trainaug) and processed CSVs are available to examination committee members via the shared Dropbox folder. Additional data available from the corresponding author on request.

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
