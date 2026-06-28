# MooAnalytica — Examination Committee Data Package

**Thesis:** An AI-Driven Behavioural Perception and Nutrition Modelling Framework as the Behavioural State Layer of a Dairy Cattle Digital Twin  
**Author:** Shreya Rao, MCS Candidate, Dalhousie University  
**Supervisor:** Dr. Suresh Neethirajan  
**Defence date:** July 29, 2026  
**Associated publication:** Rao, Garcia & Neethirajan (2026), *npj Veterinary Sciences* 1:3. DOI: 10.1038/s44433-026-00004-x

This folder contains all materials required to understand, verify, and reproduce the results reported in the thesis. It is shared exclusively with the examination committee for internal review and is not intended for public distribution.

---

## Folder Structure

```
MooAnalytica_ExaminerPackage/
│
├── annotatedTrainingDataset/          Annotated video clips (10s MP4, 1920×1080)
│   ├── train/                         Unaugmented training set — 3,343 clips, 7.5 GB
│   │   ├── Drinking/
│   │   ├── Feeding & Lying/
│   │   ├── Feeding & Standing/
│   │   ├── Lying/
│   │   ├── Ruminating & Lying/
│   │   ├── Ruminating & Standing/
│   │   └── Standing/
│   ├── train_aug/                     Augmented training set — 9,652 clips, 21.8 GB
│   │   └── (same 7 class subfolders)
│   └── val/                           Validation set — 1,119 clips, ~31.7 GB
│       └── (same 7 class subfolders)
│
├── processedData/                     Processed tabular datasets for nutrition modelling
│   ├── feeding_behaviour_153days.csv  Per cow-day: feeding_min, FCM, BW, DIM, DMI (16 cows, 153 days)
│   └── milk_yield_march2025.xls       Raw milking system export (March 2025, RAC Truro NS)
│
└── modelWeights/                      Trained model checkpoints
    ├── best.pt                        YOLOv11 detection weights (19 MB)
    └── last.pt                        TimeSformer fine-tuned weights (1.46 GB)
                                       Also at: huggingface.co/shreyayayay/timesformer-dairy-cows
```

---

## Annotated Dataset

| Split | Clips | Size | Notes |
|-------|-------|------|-------|
| `train` | 3,343 | 7.5 GB | Unaugmented base training set |
| `train_aug` | 9,652 | 21.8 GB | Augmented — flip, brightness/contrast jitter, temporal jitter |
| `val` | 1,119 | ~31.7 GB | Held-out, never seen during training |
| **Total** | **14,114** | **~61 GB** | |

**Seven behaviour classes (subfolders):**

| Class | Description |
|-------|-------------|
| Drinking | At water trough |
| Feeding & Lying | Active feeding, lying posture |
| Feeding & Standing | Active feeding at bunk, standing |
| Lying | Resting, no feeding or rumination |
| Ruminating & Lying | Jaw movement, lying |
| Ruminating & Standing | Jaw movement, standing |
| Standing | Stationary, not feeding |

**Clip format:** `.mp4`, 10-second clips at 1920×1080, extracted from overhead CCTV footage  
**Source:** RAC Truro NS, March 2025, 3 overhead cameras, 17 Holstein dairy cows  
**Filename convention:** `{cam_id}_{datetime}_{session}_{row}_{timestamp}_cow_{id}.mp4`

---

## Processed Data

`feeding_behaviour_153days.csv` — primary dataset for Chapter 6 nutrition modelling (16 cows, 153 cow-days, Cow 403 excluded due to incomplete records).

| Column | Description |
|--------|-------------|
| `cow_id` | Animal ID |
| `date` | Date of observation (March 2025) |
| `feeding_min` | Daily feeding duration (min/day) from behavioural pipeline |
| `FCM_kg` | Fat-Corrected Milk (kg/day): 0.4×milk + 15×(milk×fat%/100) — NRC 2001 |
| `BW_kg` | Body weight (kg) |
| `DIM` | Days in milk |
| `DMI_kg` | Dry matter intake (kg/day) from feed monitoring system |
| `parity` | Lactation number |

---

## Model Weights

| File | Model | Size | Notes |
|------|-------|------|-------|
| `best.pt` | YOLOv11 (cow detection) | 19 MB | Place at `Models/best.pt` in repo |
| `last.pt` | TimeSformer (behaviour classification) | 1.46 GB | Set `MODEL_DIR` in `Pipeline/pipeline.py` |

---

## Reproduction Instructions

All code: **https://github.com/raoshreya17/dairycow-digitaltwin**

### Behaviour classification
1. Place `last.pt` at the path set in `Pipeline/pipeline.py`
2. Run `Pipeline/Complete_pipeline_with_TimeSformer.ipynb`
3. For robustness evaluation: `Evaluation/RobustnessDayNight.ipynb` (needs `val/` + `ts_y_pred.npy`)

### Nutrition modelling
1. Place `feeding_behaviour_153days.csv` at path set in `Nutrition/NutritionModel.ipynb`
2. Run all cells — expected: Km=30, MAPE=3.2%, R²=0.770, LOCO MAPE=5.8%, R²=0.313

### Controller simulation
1. Run `Nutrition/controller_simulation_final.ipynb` — fully self-contained, no data files needed
2. Expected: 100% convergence within 14 days, median recovery 3.0 days (500 runs)

---

## Software Dependencies

```
python >= 3.10
torch >= 2.0
ultralytics          # YOLOv11
transformers==4.41.2 # TimeSformer
supervision          # ByteTrack
decord==0.6.0
numpy, pandas, scipy, matplotlib, scikit-learn
```

Full environment: `environment.yml` in the GitHub repository.

---

## Contact

**Shreya Rao** — shreya.rao@dal.ca  
**Dr. Suresh Neethirajan** (Supervisor) — suresh.neethirajan@dal.ca  
Dalhousie University, Faculty of Agriculture, Truro NS
