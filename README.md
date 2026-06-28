# digital-twin-dairycow

AI-driven digital twin of dairy cows integrating behaviour, nutrition, and closed-loop feeding management. Developed at the MooAnalytica research group, Dalhousie University.

## Overview

The system processes continuous barn video through a four-stage pipeline:

1. **Detection** — YOLOv11 detects individual Holstein cows in each frame (mAP@50 = 0.994)
2. **Tracking** — ByteTrack with stall zone anchoring maintains per-cow identity across sessions
3. **Classification** — TimeSformer classifies seven behaviour classes at 85.0% accuracy, 22.6 fps
4. **Nutrition + Control** — DMI estimation and proportional feedback controller recommend daily feed allocation per cow

## Repository Structure

```
digital-twin-dairycow/
├── Detection/          YOLOv11 training and inference (custom Holstein dataset, 2308 images)
├── Tracking/           ByteTrack with stall zone anchoring for identity persistence
├── Classification/     SlowFast and TimeSformer training, evaluation, and comparison
├── TimeSformer 2 head/ Two-head TimeSformer variant
├── Explainability/     Attention map visualisation and GradCAM
├── Pipeline/           End-to-end inference: detection → tracking → classification → CSV output
├── Nutrition/          DMI estimation models, closed-loop controller, simulation
│   ├── nutrition_modelling.py     Full Nutrition Modelling analysis (linear, regression, M-M models)
│   ├── simulate.py                 14-day controller simulation 
│   └── dairy_feed_simulation.py    Downstream feed impact scenarios
└── best.pt             YOLOv11 model weights (trained on RAC Holstein dataset)
```

## Pipeline

```
CCTV video (1920x1080, 24 fps)
    |
    v
YOLOv11 detection  (best.pt)
    |  bounding boxes + confidence
    v
ByteTrack + stall zone anchoring
    |  cow_id, tracklet, per-frame bbox
    v
TimeSformer classification  (10s clips, 224x224)
    |  {cow_id, behaviour, start, end, duration, confidence}
    v
CSV behavioural timelines
    |  daily feeding_min, rumination_min, lying_min per cow
    v
 DMI estimation + NRC reference
    |  DMI_est(t), DMI_NRC(t), e(t)
    v
Proportional feedback controller
    |  u(t+1) = clip(u(t) + K * e(t), 15, 28)  kg DM/day
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
