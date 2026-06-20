"""
End-to-end behaviour recognition pipeline for dairy cattle digital twin.

Stages:
  1. YOLOv11 detection  — localises individual cows per frame
  2. ByteTrack tracking — maintains per-cow identity across frames
  3. TimeSformer classification — classifies seven behaviour classes
     from sliding-window clips (10 s, 50% overlap, majority smoothing)
  4. CSV output — event log and per-cow daily activity totals

Seven behaviour classes:
  0  Standing
  1  Lying
  2  Drinking
  3  Feeding & Standing
  4  Feeding & Lying
  5  Ruminating & Standing
  6  Ruminating & Lying

Activity totals (cow_activity_totals.csv) are the direct input to the nutrition estimation module (Nutrition/chapter6_mm_complete.py).

"""

import os
import time
import math
import json
import gc
from pathlib import Path
from collections import defaultdict, deque

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from transformers import TimesformerForVideoClassification, AutoConfig

# =============================================================================
# CONFIG
# =============================================================================

VIDEO_PATH          = "/content/cows.mp4"
OUT_DIR             = Path("/content/cow_logs")
WRITE_ANNOTATED_MP4 = True

# YOLOv11 / ByteTrack
YOLO_WEIGHTS        = "/content/best.pt"
YOLO_IMGSZ          = 960
YOLO_CONF           = 0.60
YOLO_IOU            = 0.50
TRACKER_YAML        = "bytetrack.yaml"

# TimeSformer
MODEL_DIR           = "/content/drive/MyDrive/Models/timesformer-cows2"
CLF_RES             = 224      # crop resolution fed to TimeSformer (112 for speed)
WINDOW_FRAMES       = 12       # T: frames per classification clip
SAMPLE_FPS          = 1.6      # effective sampling rate (16 frames per 10 s)
WINDOW_OVERLAP      = 0.5      # classification every 5 s at 1.6 fps
SMOOTH_K            = 3        # majority vote over last K predictions
MIN_WARMUP_FRAMES   = 8        # minimum frames before first classification

# Segment filtering
INACTIVITY_TIMEOUT_S = 10
MIN_SEGMENT_S        = 2

# Crop padding target before resize (matches training preprocessing)
PAD_TARGET_HW = (640, 640)

# Default label map — overridden by checkpoint config if present
ID2LABEL = {
    0: "Standing",
    1: "Lying",
    2: "Drinking",
    3: "Feeding & Standing",
    4: "Feeding & Lying",
    5: "Ruminating & Standing",
    6: "Ruminating & Lying",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def sec_to_hms(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def center_pad_bbox_crop(img_rgb, xyxy, pad_target=(640, 640), out_size=224):
    """
    Crop the YOLO bounding box from img_rgb, scale down if larger than
    pad_target, center-pad to pad_target, and resize to out_size.
    Returns RGB uint8 array or None if the crop is empty.
    """
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    th, tw = pad_target
    h, w = crop.shape[:2]
    if h > th or w > tw:
        scale = min(th / h, tw / w)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = nh, nw

    top    = (th - h) // 2
    bottom = th - h - top
    left   = (tw - w) // 2
    right  = tw - w - left
    padded = cv2.copyMakeBorder(
        crop, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    if out_size and out_size != th:
        padded = cv2.resize(padded, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return padded


IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_clip(frames_hw3):
    """
    Convert list of T RGB (H, W, 3) uint8 frames to a normalised
    (1, T, C, H, W) float32 tensor for TimeSformer.
    """
    arr = np.stack(frames_hw3, axis=0).astype(np.float32) / 255.0
    arr = (arr - IMNET_MEAN) / IMNET_STD
    arr = arr.transpose(0, 3, 1, 2)          # T, C, H, W
    return torch.from_numpy(arr).unsqueeze(0) # 1, T, C, H, W


def majority(lst):
    if not lst:
        return None
    return max(set(lst), key=lst.count)


# =============================================================================
# LOAD MODELS
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

yolo = YOLO(YOLO_WEIGHTS)
try:
    yolo.fuse()
except Exception as e:
    print(f"YOLO fuse skipped: {e}")

cfg = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
try:
    if getattr(cfg, "id2label", None):
        ID2LABEL = {int(k): v for k, v in cfg.id2label.items()}
        print(f"Loaded id2label from checkpoint: {ID2LABEL}")
    else:
        print("No id2label in config; using default map.")
except Exception as e:
    print(f"id2label read failed, using default map. Detail: {e}")

clf = TimesformerForVideoClassification.from_pretrained(
    MODEL_DIR, config=cfg, ignore_mismatched_sizes=True
).to(device).eval()

print("Models loaded.")

# =============================================================================
# MAIN PIPELINE
# =============================================================================

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Cannot open video: {VIDEO_PATH}"
video_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
video_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

sample_stride  = max(1, int(round(video_fps / max(0.1, SAMPLE_FPS))))
step_frames    = max(1, int(round(WINDOW_FRAMES * (1.0 - WINDOW_OVERLAP))))
timeout_frames = int(round(INACTIVITY_TIMEOUT_S * video_fps))

print(f"Video: {video_fps:.2f} fps  {frame_count} frames  {video_w}x{video_h}")
print(f"Sample stride: {sample_stride}  Window: {WINDOW_FRAMES}  Step: {step_frames}")

# Per-track state
buffers             = defaultdict(lambda: deque(maxlen=WINDOW_FRAMES))
last_sample_frame   = defaultdict(lambda: -(10**9))
last_classify_frame = defaultdict(lambda: -(10**9))
last_seen_frame     = dict()
pred_hist           = defaultdict(lambda: deque(maxlen=SMOOTH_K))
active_event        = dict()
events              = []

writer = None
if WRITE_ANNOTATED_MP4:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUT_DIR / "annotated.mp4"), fourcc, video_fps, (video_w, video_h)
    )

start_time = time.time()
frame_idx  = -1

gen = yolo.track(
    source=VIDEO_PATH, stream=True, imgsz=YOLO_IMGSZ,
    conf=YOLO_CONF, iou=YOLO_IOU, tracker=TRACKER_YAML,
    device=0 if device == "cuda" else "cpu",
    verbose=False, persist=True
)

DEBUG_DUMPS = 3
dbg_dumped  = 0

for res in tqdm(gen, desc="Processing", total=frame_count):
    frame_idx += 1
    img_bgr = res.orig_img
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    xyxy    = None
    ids_arr = None

    if res.boxes is not None and res.boxes.xyxy is not None:
        xyxy    = res.boxes.xyxy.cpu().numpy()
        ids     = res.boxes.id
        ids_arr = (ids.cpu().numpy().astype(int)
                   if ids is not None
                   else np.arange(len(xyxy), dtype=int))

        for bb, tid in zip(xyxy, ids_arr):
            last_seen_frame[tid] = frame_idx

            # Sample crop into buffer at SAMPLE_FPS
            if frame_idx - last_sample_frame[tid] >= sample_stride:
                crop = center_pad_bbox_crop(img_rgb, bb,
                                            pad_target=PAD_TARGET_HW,
                                            out_size=CLF_RES)
                if crop is not None:
                    buffers[tid].append(crop)
                    last_sample_frame[tid] = frame_idx

            n_samples  = len(buffers[tid])
            ready_full = n_samples >= WINDOW_FRAMES
            ready_warm = (active_event.get(tid) is None) and (n_samples >= MIN_WARMUP_FRAMES)
            step_ok    = (frame_idx - last_classify_frame[tid]) >= (sample_stride * step_frames)

            if (ready_full or ready_warm) and step_ok:
                clip = list(buffers[tid])
                if n_samples < WINDOW_FRAMES:
                    # pad short clips by repeating the last frame
                    clip = clip + [clip[-1]] * (WINDOW_FRAMES - n_samples)

                pixel_values = preprocess_clip(clip).to(device)
                with torch.no_grad():
                    logits   = clf(pixel_values=pixel_values).logits
                    pred_idx = int(torch.argmax(logits, dim=-1).item())
                    label    = ID2LABEL.get(pred_idx, str(pred_idx))

                pred_hist[tid].append(label)
                smooth_label = majority(list(pred_hist[tid]))
                last_classify_frame[tid] = frame_idx

                # Save debug tile grid for first few classifications
                if dbg_dumped < DEBUG_DUMPS:
                    T    = len(clip); cols = 4; rows = math.ceil(T / cols)
                    padf = np.zeros_like(clip[0], dtype=np.uint8)
                    tiles = clip + [padf] * (rows * cols - T)
                    grid = np.concatenate(
                        [np.concatenate(tiles[r*cols:(r+1)*cols], 1) for r in range(rows)], 0
                    )
                    dbg_path = (OUT_DIR /
                                f"dbg_cow{tid}_f{frame_idx}"
                                f"_raw-{label.replace(' ','_')}"
                                f"_sm-{smooth_label.replace(' ','_')}.jpg")
                    cv2.imwrite(str(dbg_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                    dbg_dumped += 1

                # Segment boundary detection around window centre
                window_centre = frame_idx - (WINDOW_FRAMES // 2)
                if tid not in active_event:
                    active_event[tid] = {
                        "label":       smooth_label,
                        "start_frame": max(0, window_centre),
                    }
                elif smooth_label != active_event[tid]["label"]:
                    st = active_event[tid]["start_frame"]
                    et = max(st, window_centre)
                    if (et - st) / video_fps >= MIN_SEGMENT_S:
                        events.append({
                            "cow_id":      int(tid),
                            "activity":    active_event[tid]["label"],
                            "start_frame": st,
                            "end_frame":   et,
                        })
                    active_event[tid] = {
                        "label":       smooth_label,
                        "start_frame": window_centre,
                    }

    # Close tracks that have been inactive for INACTIVITY_TIMEOUT_S seconds
    to_close = [tid for tid, lastf in last_seen_frame.items()
                if frame_idx - lastf >= timeout_frames]
    for tid in to_close:
        if tid in active_event:
            st = active_event[tid]["start_frame"]
            et = max(st, last_seen_frame[tid])
            if (et - st) / video_fps >= MIN_SEGMENT_S:
                events.append({
                    "cow_id":      int(tid),
                    "activity":    active_event[tid]["label"],
                    "start_frame": st,
                    "end_frame":   et,
                })
        buffers.pop(tid, None)
        pred_hist.pop(tid, None)
        active_event.pop(tid, None)
        last_seen_frame.pop(tid, None)

    # Write annotated frame
    if WRITE_ANNOTATED_MP4 and xyxy is not None:
        for bb, tid in zip(xyxy, ids_arr):
            lab = active_event.get(int(tid), {}).get("label", "estimating")
            x1, y1, x2, y2 = map(int, bb)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{int(tid)}: {lab}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(img_bgr)

# Flush any open segments at end of video
for tid, ev in list(active_event.items()):
    st = ev["start_frame"]
    et = max(st, frame_idx)
    if (et - st) / video_fps >= MIN_SEGMENT_S:
        events.append({
            "cow_id":      int(tid),
            "activity":    ev["label"],
            "start_frame": st,
            "end_frame":   et,
        })

if writer is not None:
    writer.release()

elapsed = time.time() - start_time
print(f"Processed {frame_idx + 1} frames in {elapsed:.1f}s "
      f"({(frame_idx + 1) / max(1.0, elapsed):.1f} fps)")

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_and_summarize(events, out_dir, fps):
    if not events:
        print("No segments produced.")
        return None, None

    df = pd.DataFrame(events)
    df["start_sec"]    = df["start_frame"] / fps
    df["end_sec"]      = df["end_frame"] / fps
    df["duration_s"]   = df["end_sec"] - df["start_sec"]
    df["start_hms"]    = df["start_sec"].map(sec_to_hms)
    df["end_hms"]      = df["end_sec"].map(sec_to_hms)
    df["duration_min"] = df["duration_s"] / 60.0

    csv_events = out_dir / "cow_activity_events.csv"
    df.sort_values(["cow_id", "start_frame"]).to_csv(csv_events, index=False)
    print(f"Event log:     {csv_events}")

    agg = (df.groupby(["cow_id", "activity"])["duration_s"]
             .sum().reset_index())
    agg["duration_min"] = agg["duration_s"] / 60.0

    csv_totals = out_dir / "cow_activity_totals.csv"
    agg.sort_values(["cow_id", "duration_s"], ascending=[True, False]).to_csv(
        csv_totals, index=False)
    print(f"Activity totals: {csv_totals}")

    print("\nSummary (top 3 activities per cow):")
    view = (agg.sort_values(["cow_id", "duration_s"], ascending=[True, False])
               .groupby("cow_id").head(3))
    for _, row in view.iterrows():
        print(f"  Cow {int(row['cow_id'])}: {row['activity']}  "
              f"{row['duration_min']:.1f} min")

    return csv_events, csv_totals


save_and_summarize(events, OUT_DIR, video_fps)
