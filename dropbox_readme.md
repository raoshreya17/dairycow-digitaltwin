**Thesis:** An AI-Driven Behavioural Perception and Nutrition Modelling Framework as the Behavioural State Layer of a Dairy Cattle Digital Twin  

This folder contains all annotated data.

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
│   └── val/                           Validation set - 1,119 clips
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
| Drinking | Face in water trough |
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
