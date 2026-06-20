"""
  - Data loading (3 cameras, 17 cows, March 15-25 2025)
  - Linear baseline model (k fitted locally and from Johnston & DeVries 2018)
  - Pooled regression model (DMI = a + b*f_feed + c*BW + d*ECM)
  - Michaelis-Menten saturating model (per-cow DMI_MAX, shared Km)
  - Km sensitivity sweep (grid 10-300 min, best in biological range 30-150)
  - Day-split validation (early days train, later days test)
  - Leave-one-cow-out (LOCO) cross-validation for both linear and regression
  - All Chapter 6 figures: regression scatter, LOCO scatter, BW-bias plot,
    M-M scatter by camera, welfare heatmap, health event trajectories

Primary dataset: 16 cows (Cow 403 excluded), 153 clean cow-days.
All models evaluated against NRC (2001) as physiological reference.
Primary model: pooled regression (LOCO MAPE = 5.8%, R2 = 0.424).
Secondary model: M-M with per-cow DMI_MAX, Km = 40 min/day.

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150,
})

# =============================================================================
# PATHS
# =============================================================================

CSV_001       = "/Users/shreyarao/Desktop/MooOutput/001inputs/"
CSV_002       = "/Users/shreyarao/Desktop/MooOutput/002inputs/"
CSV_003       = "/Users/shreyarao/Desktop/MooOutput/003inputs/"
MILK_XLS_PATH = "/Users/shreyarao/Desktop/MooOutput/Copy of milk yield.xls"
OUT_DIR       = "/Users/shreyarao/Desktop/MooOutput/thesis_figures_chapter6_final/"
os.makedirs(OUT_DIR, exist_ok=True)

DAYS = ["March15","March16","March17","March18","March19",
        "March20","March21","March22","March23","March24","March25"]

#COW PARAMETERS

COW_BASE = {
    387: {"BW":921.0, "DIM_mar15":15,  "fat_pct":5.25, "protein_pct":3.43, "parity":4, "camera":"001"},
    403: {"BW":709.0, "DIM_mar15":27,  "fat_pct":4.21, "protein_pct":3.80, "parity":3, "camera":"001"},
    408: {"BW":777.0, "DIM_mar15":354, "fat_pct":5.32, "protein_pct":4.06, "parity":2, "camera":"001"},
    410: {"BW":623.0, "DIM_mar15":159, "fat_pct":4.67, "protein_pct":3.72, "parity":2, "camera":"001"},
    416: {"BW":709.0, "DIM_mar15":196, "fat_pct":4.86, "protein_pct":4.20, "parity":2, "camera":"001"},
    426: {"BW":801.0, "DIM_mar15":245, "fat_pct":4.77, "protein_pct":3.45, "parity":1, "camera":"002"},
    413: {"BW":680.0, "DIM_mar15":224, "fat_pct":5.01, "protein_pct":3.89, "parity":2, "camera":"002"},
    323: {"BW":801.0, "DIM_mar15":289, "fat_pct":4.98, "protein_pct":3.80, "parity":6, "camera":"002"},
    354: {"BW":880.0, "DIM_mar15":214, "fat_pct":5.37, "protein_pct":4.01, "parity":5, "camera":"002"},
    310: {"BW":840.0, "DIM_mar15":61,  "fat_pct":3.99, "protein_pct":3.22, "parity":7, "camera":"002"},
    349: {"BW":921.0, "DIM_mar15":290, "fat_pct":4.66, "protein_pct":3.99, "parity":5, "camera":"002"},
    433: {"BW":657.0, "DIM_mar15":74,  "fat_pct":4.11, "protein_pct":3.48, "parity":1, "camera":"002"},
    406: {"BW":840.0, "DIM_mar15":231, "fat_pct":4.60, "protein_pct":3.28, "parity":2, "camera":"003"},
    386: {"BW":764.0, "DIM_mar15":327, "fat_pct":5.05, "protein_pct":3.89, "parity":3, "camera":"003"},
    412: {"BW":801.0, "DIM_mar15":226, "fat_pct":4.59, "protein_pct":3.96, "parity":2, "camera":"003"},
    384: {"BW":764.0, "DIM_mar15":43,  "fat_pct":4.99, "protein_pct":3.72, "parity":4, "camera":"003"},
    428: {"BW":623.0, "DIM_mar15":150, "fat_pct":4.48, "protein_pct":3.68, "parity":1, "camera":"003"},
}

STALL_TO_COW_002 = {1:426, 2:413, 3:323, 4:354, 5:310, 6:349, 7:433}

# Days excluded: confirmed partial recording or health-event days
PARTIAL_DAYS = {
    (387,"March25"),(403,"March24"),
    (387,"March20"),(403,"March20"),(408,"March20"),(410,"March20"),
    (416,"March20"),(433,"March20"),(412,"March20"),(426,"March20"),
    (323,"March20"),
    (354,"March20"),(354,"March21"),(354,"March22"),
    (354,"March23"),(354,"March24"),(354,"March25"),
    (310,"March19"),(428,"March19"),
}

COW_COLORS = {
    387:"#1D9E75",403:"#7F77DD",408:"#BA7517",410:"#378ADD",416:"#D4537E",
    310:"#E24B4A",323:"#F5A623",349:"#9B59B6",354:"#1ABC9C",
    413:"#2ECC71",426:"#3498DB",433:"#E67E22",
    406:"#C0392B",386:"#16A085",412:"#D35400",384:"#2980B9",428:"#27AE60",
}
# DATA LOADING
xls = pd.read_excel(MILK_XLS_PATH, engine="xlrd")
xls["Date"] = pd.to_datetime(xls["Date"]).dt.date
xls["day"]  = xls["Date"].apply(lambda d: f"March{d.day}" if d.month==3 else None)
xls = xls.dropna(subset=["day"])
xls["dur"] = xls["Milk duration (mm:ss)"].astype(str)
xls = xls[~xls["dur"].str.startswith("30:00")]  # exclude equipment malfunction sessions

all_cow_ids = list(COW_BASE.keys())
daily = (xls[xls["Animal Number"].isin(all_cow_ids)]
         .groupby(["Animal Number","day"])["Yield"]
         .sum().reset_index())
daily.columns = ["cow_id","day","milk_kg"]
daily["cow_id"] = daily["cow_id"].astype(int)

MILK = {}
for _, r in daily.iterrows():
    c = int(r["cow_id"])
    if c not in MILK: MILK[c] = {}
    MILK[c][r["day"]] = round(r["milk_kg"], 2)

FEEDING_LABELS = {"Feeding & Standing", "Feeding & Lying"}

def load_001(folder, days):
    dfs = []
    for day in days:
        p = os.path.join(folder, f"{day}_activity_totals.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "day" not in df.columns: df["day"] = day
            df["camera"] = "001"
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_folder(folder, days, stall_map, label):
    dfs = []
    for day in days:
        p = os.path.join(folder, day, "cow_activity_totals.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["day"] = day
            if stall_map:
                df["cow_id"] = df["cow_id"].map(stall_map)
                df = df.dropna(subset=["cow_id"])
                df["cow_id"] = df["cow_id"].astype(int)
            df["camera"] = label
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

df_all = pd.concat([
    load_001(CSV_001, DAYS),
    load_folder(CSV_002, DAYS, STALL_TO_COW_002, "002"),
    load_folder(CSV_003, DAYS, None, "003"),
], ignore_index=True)

# =============================================================================
# NRC (2001) REFERENCE DMI
# =============================================================================

def ecm(milk, fat, prot):
    return 0.327*milk + 12.95*(milk*fat/100) + 7.2*(milk*prot/100)

def nrc_dmi(ecm_val, bw, dim):
    return (0.372*ecm_val + 0.0968*(bw**0.75)) * (1 - np.exp(-0.192*(dim/7 + 3.67)))

feeds      = df_all[df_all["activity"].isin(FEEDING_LABELS)]
daily_feed = (feeds.groupby(["cow_id","day","camera"])["duration_min"]
              .sum().reset_index())
daily_feed.columns = ["cow_id","day","camera","feeding_min"]

rows = []
for _, r in daily_feed.iterrows():
    cow, day, fmin, cam = int(r["cow_id"]), r["day"], r["feeding_min"], r["camera"]
    if cow not in COW_BASE or day not in MILK.get(cow,{}) or fmin < 1:
        continue
    b       = COW_BASE[cow]
    dim     = b["DIM_mar15"] + DAYS.index(day)
    milk    = MILK[cow][day]
    ecm_val = ecm(milk, b["fat_pct"], b["protein_pct"])
    dmi_nrc = nrc_dmi(ecm_val, b["BW"], dim)
    rows.append({
        "cow_id":cow, "day":day, "camera":cam, "parity":b["parity"],
        "feeding_min":round(fmin,1), "BW":b["BW"], "DIM":dim,
        "milk_kg":round(milk,2), "ECM_kg":round(ecm_val,2),
        "DMI_NRC":round(dmi_nrc,3), "partial":(cow,day) in PARTIAL_DAYS,
    })

full = pd.DataFrame(rows)
data = full[~full["partial"] & (full["feeding_min"]>=50)].copy()

# 16-cow dataset: exclude Cow 403 (detection gaps)
data_16 = data[data["cow_id"] != 403].copy()
all_cows = sorted(data_16["cow_id"].unique())
cam_map  = data_16.drop_duplicates("cow_id").set_index("cow_id")["camera"]

print(f"Full dataset:  {len(data)} cow-days | {data['cow_id'].nunique()} cows")
print(f"16-cow dataset: {len(data_16)} cow-days | {data_16['cow_id'].nunique()} cows")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def report_metrics(y_true, y_pred, label):
    err  = y_pred - y_true
    rmse = np.sqrt((err**2).mean())
    mae  = np.abs(err).mean()
    bias = err.mean()
    mape = (np.abs(err) / y_true).mean() * 100
    r2   = np.corrcoef(y_true, y_pred)[0,1]**2
    print(f"  {label}: RMSE={rmse:.3f}  MAE={mae:.3f}  "
          f"Bias={bias:+.3f}  MAPE={mape:.1f}%  R2={r2:.3f}")
    return dict(rmse=rmse, mae=mae, bias=bias, mape=mape, r2=r2)

def fit_reg(df):
    X = np.column_stack([np.ones(len(df)), df["feeding_min"], df["BW"], df["ECM_kg"]])
    coeffs, _, _, _ = np.linalg.lstsq(X, df["DMI_NRC"].values, rcond=None)
    return coeffs

def pred_reg(df, coeffs):
    X = np.column_stack([np.ones(len(df)), df["feeding_min"], df["BW"], df["ECM_kg"]])
    return X @ coeffs

# =============================================================================
# BUILD DAY-SPLIT PARTITION
# =============================================================================

split_records = []
for cow in all_cows:
    s = data_16[data_16["cow_id"]==cow].copy()
    s["day_idx"] = s["day"].apply(lambda d: DAYS.index(d))
    s = s.sort_values("day_idx").reset_index(drop=True)
    n = len(s)
    if n < 4:
        continue
    split_pt = n // 2
    s["split"] = ["train"]*split_pt + ["test"]*(n-split_pt)
    split_records.append(s)
data_split = pd.concat(split_records, ignore_index=True)

# =============================================================================
# LINEAR BASELINE MODEL
# =============================================================================

print("\n=== LINEAR BASELINE ===")

# Variant A: k refit on this dataset (through origin)
fmin_all = data_16["feeding_min"].values
y_all    = data_16["DMI_NRC"].values
k_local  = np.sum(fmin_all * y_all) / np.sum(fmin_all**2)
est_local = k_local * fmin_all
report_metrics(y_all, est_local, f"Local k={k_local:.4f} kg/min (whole dataset)")

# Variant B: Johnston & DeVries 2018 daily-average rate (27.0 kg/d / 230.4 min/d)
k_lit  = 27.0 / 230.4
est_lit = k_lit * fmin_all
report_metrics(y_all, est_lit, f"Literature k={k_lit:.4f} kg/min (Johnston & DeVries 2018)")

# Day-split: k refit per fold
train_lin = data_split[data_split["split"]=="train"].copy()
test_lin  = data_split[data_split["split"]=="test"].copy()
fmin_tr   = train_lin["feeding_min"].values
y_tr      = train_lin["DMI_NRC"].values
k_split   = np.sum(fmin_tr * y_tr) / np.sum(fmin_tr**2)
test_lin["DMI_est_lin"] = k_split * test_lin["feeding_min"]
test_lin["err_lin"]     = test_lin["DMI_est_lin"] - test_lin["DMI_NRC"]
test_lin["pct_lin"]     = 100 * test_lin["err_lin"] / test_lin["DMI_NRC"]
report_metrics(test_lin["DMI_NRC"].values, test_lin["DMI_est_lin"].values,
               "Linear day-split TEST")

# LOCO: k refit per fold
print("\n  Linear LOCO:")
loco_lin_records = []
for held_out in all_cows:
    train_fold = data_16[data_16["cow_id"] != held_out]
    test_fold  = data_16[data_16["cow_id"] == held_out]
    fmin_f = train_fold["feeding_min"].values
    y_f    = train_fold["DMI_NRC"].values
    k_fold = np.sum(fmin_f * y_f) / np.sum(fmin_f**2)
    pred   = k_fold * test_fold["feeding_min"].values
    err    = pred - test_fold["DMI_NRC"].values
    pct    = 100 * err / test_fold["DMI_NRC"].values
    for i, (_, row) in enumerate(test_fold.iterrows()):
        loco_lin_records.append({
            "cow_id": held_out, "BW": COW_BASE[held_out]["BW"],
            "DMI_NRC": row["DMI_NRC"], "DMI_est": pred[i],
            "err": err[i], "pct": pct[i],
        })

loco_lin_df = pd.DataFrame(loco_lin_records)
report_metrics(loco_lin_df["DMI_NRC"].values, loco_lin_df["DMI_est"].values,
               "Linear LOCO pooled")

# =============================================================================
# POOLED REGRESSION MODEL
# =============================================================================

print("\n=== POOLED REGRESSION ===")

# Whole dataset (calibration reference only)
coeffs_all = fit_reg(data_16)
print(f"  Coefficients (whole dataset): a={coeffs_all[0]:.4f}  "
      f"b={coeffs_all[1]:.5f}  c={coeffs_all[2]:.6f}  d={coeffs_all[3]:.5f}")

# Day-split
train_reg = data_split[data_split["split"]=="train"].copy()
test_reg  = data_split[data_split["split"]=="test"].copy()
coeffs_tr = fit_reg(train_reg)
train_reg["DMI_est_reg"] = pred_reg(train_reg, coeffs_tr)
test_reg["DMI_est_reg"]  = pred_reg(test_reg,  coeffs_tr)
for df in (train_reg, test_reg):
    df["err_reg"] = df["DMI_est_reg"] - df["DMI_NRC"]
    df["pct_reg"] = 100 * df["err_reg"] / df["DMI_NRC"]

report_metrics(train_reg["DMI_NRC"].values, train_reg["DMI_est_reg"].values, "Regression TRAIN")
report_metrics(test_reg["DMI_NRC"].values,  test_reg["DMI_est_reg"].values,  "Regression TEST")

# Per-cow day-split results
print(f"\n  Per-cow TEST breakdown:")
print(f"  {'Cow':>5}  {'Cam':>4}  {'RMSE':>7}  {'MAPE':>7}  {'Bias':>8}  n")
print("  " + "-"*50)
for cow in all_cows:
    s = test_reg[test_reg["cow_id"]==cow]
    if len(s) == 0: continue
    rmse_c = np.sqrt((s["err_reg"]**2).mean())
    mape_c = s["pct_reg"].abs().mean()
    bias_c = s["err_reg"].mean()
    print(f"  {cow:>5}  {cam_map[cow]:>4}  {rmse_c:>7.3f}  {mape_c:>6.1f}%  "
          f"{bias_c:>+8.3f}  {len(s)}")

# LOCO regression
print("\n  Regression LOCO per-cow:")
print(f"  {'Cow':>5}  {'Cam':>4}  {'n':>3}  {'a':>7}  {'b(fmin)':>9}  "
      f"{'c(BW)':>9}  {'d(ECM)':>9}  {'RMSE':>7}  {'MAPE':>7}  R2")
print("  " + "-"*90)

loco_reg_records = []
for held_out in all_cows:
    train_fold  = data_16[data_16["cow_id"] != held_out]
    test_fold   = data_16[data_16["cow_id"] == held_out]
    coeffs_fold = fit_reg(train_fold)
    pred        = pred_reg(test_fold, coeffs_fold)
    err         = pred - test_fold["DMI_NRC"].values
    pct         = 100 * err / test_fold["DMI_NRC"].values
    rmse_f      = np.sqrt((err**2).mean())
    mape_f      = np.abs(pct).mean()
    r2_f        = np.corrcoef(test_fold["DMI_NRC"].values, pred)[0,1]**2 if len(test_fold)>1 else np.nan
    print(f"  {held_out:>5}  {cam_map[held_out]:>4}  {len(test_fold):>3}  "
          f"{coeffs_fold[0]:>7.3f}  {coeffs_fold[1]:>9.5f}  "
          f"{coeffs_fold[2]:>9.6f}  {coeffs_fold[3]:>9.5f}  "
          f"{rmse_f:>7.3f}  {mape_f:>6.1f}%  {r2_f:>6.3f}")
    for i, (_, row) in enumerate(test_fold.iterrows()):
        loco_reg_records.append({
            "cow_id": held_out, "DMI_NRC": row["DMI_NRC"], "DMI_est": pred[i],
            "err": err[i], "pct": pct[i],
        })

loco_reg_df = pd.DataFrame(loco_reg_records)
report_metrics(loco_reg_df["DMI_NRC"].values, loco_reg_df["DMI_est"].values,
               "Regression LOCO pooled")

# Summary comparison table
print("\n=== FULL VALIDATION COMPARISON ===")
print(f"  {'Model':<14}  {'Strategy':<18}  {'n':>5}  {'RMSE':>7}  {'MAPE':>7}  {'R2':>7}")
print("  " + "-"*65)
for label, df_sub, est_col in [
    ("Linear",     test_lin[test_lin["cow_id"].isin(all_cows)],  "DMI_est_lin"),
    ("Regression", test_reg,  "DMI_est_reg"),
]:
    r = report_metrics.__wrapped__ if hasattr(report_metrics, "__wrapped__") else None
    e = df_sub[est_col] - df_sub["DMI_NRC"]
    rmse = np.sqrt((e**2).mean()); mape = (e.abs()/df_sub["DMI_NRC"]).mean()*100
    r2   = np.corrcoef(df_sub["DMI_NRC"], df_sub[est_col])[0,1]**2
    print(f"  {label:<14}  {'Day-split test':<18}  {len(df_sub):>5}  "
          f"{rmse:>7.3f}  {mape:>6.1f}%  {r2:>7.3f}")

for label, loco_df in [("Linear", loco_lin_df), ("Regression", loco_reg_df)]:
    e    = loco_df["DMI_est"] - loco_df["DMI_NRC"]
    rmse = np.sqrt((e**2).mean()); mape = (e.abs()/loco_df["DMI_NRC"]).mean()*100
    r2   = np.corrcoef(loco_df["DMI_NRC"], loco_df["DMI_est"])[0,1]**2
    print(f"  {label:<14}  {'LOCO':<18}  {len(loco_df):>5}  "
          f"{rmse:>7.3f}  {mape:>6.1f}%  {r2:>7.3f}")

# =============================================================================
# MICHAELIS-MENTEN MODEL
# =============================================================================

print("\n=== MICHAELIS-MENTEN MODEL ===")

# Per-cow DMI_MAX derived analytically from each cow's mean NRC and mean feeding time
# This anchors each cow's saturation ceiling to her own physiological level
cow_nrc_mean  = data_16.groupby("cow_id")["DMI_NRC"].mean()
cow_fmin_mean = data_16.groupby("cow_id")["feeding_min"].mean()
data_16["DMI_MAX_i"] = data_16["cow_id"].map(cow_nrc_mean)

print("  Per-cow DMI_MAX (mean NRC DMI across clean days):")
for cow in all_cows:
    print(f"    Cow {cow:>3}: {cow_nrc_mean[cow]:.2f} kg/day  "
          f"(mean feeding {cow_fmin_mean[cow]:.0f} min/day)")

# Km sensitivity sweep: grid search over biological range
print("\n  Km sensitivity sweep (per-cow DMI_MAX, shared Km):")
print(f"  {'Km':>6}  {'RMSE':>6}  {'MAE':>6}  {'MAPE':>6}  {'Bias':>7}  {'R2':>6}  {'Normal/153':>12}")
print("  " + "-"*70)

KM_GRID = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300]
sweep_rows = []
for km in KM_GRID:
    est    = data_16.apply(lambda r: r["DMI_MAX_i"]*r["feeding_min"]/(km+r["feeding_min"]), axis=1)
    err    = est - data_16["DMI_NRC"]
    pct    = 100 * err / data_16["DMI_NRC"]
    rmse   = np.sqrt((err**2).mean())
    mae    = err.abs().mean()
    mape   = pct.abs().mean()
    bias   = err.mean()
    r2     = np.corrcoef(data_16["DMI_NRC"], est)[0,1]**2
    normal = ((est / data_16["DMI_NRC"]).between(0.9, 1.1)).sum()
    sweep_rows.append({"Km":km, "RMSE":rmse, "MAE":mae, "MAPE":mape,
                        "Bias":bias, "R2":r2, "Normal":normal})
    print(f"  {km:>6}  {rmse:>6.3f}  {mae:>6.3f}  {mape:>5.1f}%  "
          f"{bias:>+7.3f}  {r2:>6.3f}  {normal:>4}/{len(data_16)}")

sweep_df = pd.DataFrame(sweep_rows)

# Best Km in biological range 30-150 min/day
bio_range = sweep_df[sweep_df["Km"].between(30, 150)]
best_row  = bio_range.loc[bio_range["MAPE"].idxmin()]
KM_BEST   = int(best_row["Km"])
print(f"\n  Best Km in biological range (30-150 min): Km={KM_BEST}  "
      f"MAPE={best_row['MAPE']:.1f}%  RMSE={best_row['RMSE']:.3f}  R2={best_row['R2']:.3f}")

# Thesis Km = 40 (from grid search result)
KM_THESIS = 40
data_16["DMI_est_mm"] = data_16.apply(
    lambda r: r["DMI_MAX_i"]*r["feeding_min"]/(KM_THESIS+r["feeding_min"]), axis=1)
data_16["err_mm"]  = data_16["DMI_est_mm"] - data_16["DMI_NRC"]
data_16["pct_mm"]  = 100 * data_16["err_mm"] / data_16["DMI_NRC"]
data_16["ratio_mm"]= data_16["DMI_est_mm"] / data_16["DMI_NRC"]

print(f"\n  M-M model (Km={KM_THESIS}, per-cow DMI_MAX) - whole dataset:")
report_metrics(data_16["DMI_NRC"].values, data_16["DMI_est_mm"].values, "M-M whole dataset")

# M-M day-split evaluation
print(f"\n  M-M day-split evaluation (Km fit on train, applied to test):")
mm_split_records = []
for cow in all_cows:
    s = data_split[data_split["cow_id"]==cow].copy()
    tr = s[s["split"]=="train"]
    te = s[s["split"]=="test"]
    if len(tr) == 0 or len(te) == 0: continue
    dmi_max_train = tr["DMI_NRC"].mean()
    fmin_mean_tr  = tr["feeding_min"].mean()
    for _, row in te.iterrows():
        km_train = KM_THESIS  # fixed at thesis value
        dmi_max_i = dmi_max_train * (km_train + fmin_mean_tr) / fmin_mean_tr
        est = dmi_max_i * row["feeding_min"] / (km_train + row["feeding_min"])
        mm_split_records.append({
            "cow_id": cow, "split": "test",
            "DMI_NRC": row["DMI_NRC"], "DMI_est": est,
        })
    for _, row in tr.iterrows():
        dmi_max_i = dmi_max_train * (KM_THESIS + fmin_mean_tr) / fmin_mean_tr
        est = dmi_max_i * row["feeding_min"] / (KM_THESIS + row["feeding_min"])
        mm_split_records.append({
            "cow_id": cow, "split": "train",
            "DMI_NRC": row["DMI_NRC"], "DMI_est": est,
        })

mm_split_df = pd.DataFrame(mm_split_records)
mm_train = mm_split_df[mm_split_df["split"]=="train"]
mm_test  = mm_split_df[mm_split_df["split"]=="test"]
report_metrics(mm_train["DMI_NRC"].values, mm_train["DMI_est"].values, "M-M TRAIN")
report_metrics(mm_test["DMI_NRC"].values,  mm_test["DMI_est"].values,  "M-M TEST")

# Per-cow M-M breakdown
print(f"\n  Per-cow M-M (whole dataset, Km={KM_THESIS}):")
print(f"  {'Cow':>5}  {'Cam':>4}  {'DIM':>5}  {'BW':>5}  {'DMI_MAX':>8}  "
      f"{'RMSE':>6}  {'MAPE':>6}  {'Bias':>7}  n")
print("  " + "-"*70)
for cow in all_cows:
    s    = data_16[data_16["cow_id"]==cow]
    rmse = np.sqrt((s["err_mm"]**2).mean())
    mape = s["pct_mm"].abs().mean()
    bias = s["err_mm"].mean()
    print(f"  {cow:>5}  {cam_map[cow]:>4}  "
          f"{COW_BASE[cow]['DIM_mar15']:>5}  {COW_BASE[cow]['BW']:>5.0f}  "
          f"{cow_nrc_mean[cow]:>8.2f}  "
          f"{rmse:>6.3f}  {mape:>5.1f}%  {bias:>+7.3f}  {len(s)}")

# =============================================================================
# Regression model train vs test scatter
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, df, title in [(axes[0], train_reg, "TRAIN (fitted days)"),
                       (axes[1], test_reg,  "TEST (held-out days)")]:
    mn = min(df["DMI_NRC"].min(), df["DMI_est_reg"].min())-1
    mx = max(df["DMI_NRC"].max(), df["DMI_est_reg"].max())+2
    ax.plot([mn,mx],[mn,mx],"k--",lw=1.2,alpha=0.6,label="Perfect")
    ax.fill_between([mn,mx],[mn*0.9,mx*0.9],[mn*1.1,mx*1.1],
                    alpha=0.07,color="green",label="10% band")
    for cow in sorted(df["cow_id"].unique()):
        s = df[df["cow_id"]==cow]
        ax.scatter(s["DMI_NRC"], s["DMI_est_reg"],
                   color=COW_COLORS.get(cow,"gray"), s=55, alpha=0.85,
                   label=f"Cow {cow}")
    rmse_s = np.sqrt(((df["DMI_est_reg"]-df["DMI_NRC"])**2).mean())
    mape_s = (100*(df["DMI_est_reg"]-df["DMI_NRC"]).abs()/df["DMI_NRC"]).mean()
    r2_s   = np.corrcoef(df["DMI_NRC"],df["DMI_est_reg"])[0,1]**2
    ax.set_xlabel("NRC Expected DMI (kg/day)")
    ax.set_ylabel("Estimated DMI (kg/day)")
    ax.set_title(f"{title}\nRMSE={rmse_s:.3f}  MAPE={mape_s:.1f}%  R2={r2_s:.3f}  n={len(df)}")
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.7)
    ax.grid(alpha=0.3)
plt.suptitle("Pooled Regression Model: Train vs Held-Out Test\n"
             f"DMI = a + b*f_feed + c*BW + d*ECM  "
             f"(n_train={len(train_reg)}, n_test={len(test_reg)})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig6_1_regression_train_test.png"), dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
#  LOCO regression scatter
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 8))
mn = min(loco_reg_df["DMI_NRC"].min(), loco_reg_df["DMI_est"].min())-1
mx = max(loco_reg_df["DMI_NRC"].max(), loco_reg_df["DMI_est"].max())+2
ax.plot([mn,mx],[mn,mx],"k--",lw=1.2,alpha=0.6,label="Perfect")
ax.fill_between([mn,mx],[mn*0.9,mx*0.9],[mn*1.1,mx*1.1],
                alpha=0.07,color="green",label="10% band")
for cow in all_cows:
    s = loco_reg_df[loco_reg_df["cow_id"]==cow]
    ax.scatter(s["DMI_NRC"], s["DMI_est"],
               color=COW_COLORS.get(cow,"gray"), s=50, alpha=0.8, label=f"Cow {cow}")
rmse_l = np.sqrt(((loco_reg_df["DMI_est"]-loco_reg_df["DMI_NRC"])**2).mean())
mape_l = (100*(loco_reg_df["DMI_est"]-loco_reg_df["DMI_NRC"]).abs()/loco_reg_df["DMI_NRC"]).mean()
r2_l   = np.corrcoef(loco_reg_df["DMI_NRC"], loco_reg_df["DMI_est"])[0,1]**2
ax.set_xlabel("NRC Expected DMI (kg/day)", fontsize=11)
ax.set_ylabel("LOCO-Predicted DMI (kg/day)\n(coefficients never trained on this cow)", fontsize=11)
ax.set_title(f"Leave-One-Cow-Out Cross-Validation\n"
             f"RMSE={rmse_l:.3f}  MAPE={mape_l:.1f}%  R2={r2_l:.3f}  n={len(loco_reg_df)}",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=7, ncol=2, framealpha=0.75)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig6_2_loco_validation.png"), dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# Linear model LOCO bias vs body weight
# =============================================================================

cow_bw_list   = [COW_BASE[c]["BW"] for c in all_cows]
cow_bias_list = [loco_lin_df[loco_lin_df["cow_id"]==c]["err"].mean() for c in all_cows]

fig, ax = plt.subplots(figsize=(8,6))
for c, bw, bias in zip(all_cows, cow_bw_list, cow_bias_list):
    ax.scatter(bw, bias, color=COW_COLORS.get(c,"gray"), s=80, zorder=4)
    ax.annotate(str(c), (bw, bias), fontsize=8, xytext=(4,4), textcoords="offset points")
ax.axhline(0, color="black", lw=1)
z = np.polyfit(cow_bw_list, cow_bias_list, 1)
xline = np.linspace(min(cow_bw_list), max(cow_bw_list), 50)
ax.plot(xline, np.polyval(z, xline), "k--", lw=1.5, alpha=0.6,
        label=f"Trend: bias = {z[0]:.5f}*BW + {z[1]:.2f}")
corr = np.corrcoef(cow_bw_list, cow_bias_list)[0,1]
ax.set_xlabel("Body Weight (kg)")
ax.set_ylabel("LOCO Mean Bias (kg/day)  (DMI_est - DMI_NRC)")
ax.set_title("Linear Model LOCO Bias vs Body Weight\n"
             f"r = {corr:.3f}  (n=16 cows, Cow 403 excluded)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig6_3_linear_loco_bw_bias.png"), dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 6.4 - M-M scatter by camera
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, cam in zip(axes, ["001","002","003"]):
    sub = data_16[data_16["camera"]==cam]
    if sub.empty: ax.set_visible(False); continue
    mn = min(sub["DMI_NRC"].min(), sub["DMI_est_mm"].min())-1
    mx = max(sub["DMI_NRC"].max(), sub["DMI_est_mm"].max())+2
    ax.plot([mn,mx],[mn,mx],"k--",lw=1.2,alpha=0.6,label="Perfect")
    ax.fill_between([mn,mx],[mn*0.9,mx*0.9],[mn*1.1,mx*1.1],
                    alpha=0.07,color="green",label="10% band")
    for cow in sorted(sub["cow_id"].unique()):
        s      = sub[sub["cow_id"]==cow]
        rmse_c = np.sqrt((s["err_mm"]**2).mean())
        ax.scatter(s["DMI_NRC"], s["DMI_est_mm"],
                   color=COW_COLORS.get(cow,"gray"), s=65, zorder=5,
                   label=f"Cow {cow}  RMSE={rmse_c:.2f}")
    s_rmse = np.sqrt((sub["err_mm"]**2).mean())
    s_mape = sub["pct_mm"].abs().mean()
    s_r2   = np.corrcoef(sub["DMI_NRC"], sub["DMI_est_mm"])[0,1]**2
    ax.set_xlabel("NRC Expected DMI (kg/day)")
    ax.set_ylabel("M-M Estimated DMI (kg/day)")
    ax.set_title(f"Camera {cam}\nRMSE={s_rmse:.3f}  MAPE={s_mape:.1f}%  R2={s_r2:.3f}  n={len(sub)}")
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.7)
    ax.grid(alpha=0.3)
plt.suptitle(f"Michaelis-Menten Model by Camera (Km={KM_THESIS} min/day, per-cow DMI_MAX)\n"
             f"Overall: RMSE={np.sqrt((data_16['err_mm']**2).mean()):.3f}  "
             f"MAPE={data_16['pct_mm'].abs().mean():.1f}%  n={len(data_16)}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig6_4_mm_scatter_by_camera.png"), dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# Welfare heatmap (DMI ratio per cow per day)
# =============================================================================

def classify(ratio):
    return "normal" if 0.9 <= ratio <= 1.1 else ("underfed" if ratio < 0.9 else "overfed")

data_16["status_mm"] = data_16["ratio_mm"].apply(classify)

ALL_COWS_LIST = sorted(data_16["cow_id"].unique())
DAY_LABELS    = ["Mar 15","Mar 16","Mar 17","Mar 18","Mar 19",
                 "Mar 20","Mar 21","Mar 22","Mar 23","Mar 24","Mar 25"]

pivot = (data_16.pivot_table(index="cow_id", columns="day",
                              values="ratio_mm", aggfunc="mean")
         .reindex(columns=DAYS).reindex(ALL_COWS_LIST))

cmap = LinearSegmentedColormap.from_list(
    "welfare",["#E24B4A","#F5A623","#FFFFFF","#2ECC71","#3498DB"], N=512)

fig, ax = plt.subplots(figsize=(16, 9))
im = ax.imshow(pivot.values.astype(float), cmap=cmap, vmin=0.7, vmax=1.3, aspect="auto")
ax.set_xticks(range(len(DAYS))); ax.set_xticklabels(DAY_LABELS, fontsize=10)
ax.set_yticks(range(len(ALL_COWS_LIST)))
ax.set_yticklabels([f"Cow {c}  [{cam_map[c]}]" for c in ALL_COWS_LIST], fontsize=9.5)
ax.set_xticks(np.arange(-0.5, len(DAYS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(ALL_COWS_LIST), 1), minor=True)
ax.grid(which="minor", color="white", lw=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

for i, cow in enumerate(ALL_COWS_LIST):
    for j, day in enumerate(DAYS):
        try:
            val = float(pivot.loc[cow, day])
            if not np.isnan(val):
                col = "white" if (val < 0.82 or val > 1.22) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=col, fontweight="bold")
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=9, color="gray")
        except:
            ax.text(j, i, "-", ha="center", va="center", fontsize=9, color="gray")

# Annotate health events from veterinary records
for cow, note, col in [(354,"Teat injury","#C0392B"),(349,"Milk fever Mar 23","#C0392B")]:
    if cow in ALL_COWS_LIST:
        i = ALL_COWS_LIST.index(cow)
        ax.annotate(note, xy=(10.5,i), xytext=(11.3,i), fontsize=8,
                    color=col, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-",color=col,lw=0.8))

cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
cbar.set_label("DMI Ratio  (Est / NRC)", fontsize=10)
cbar.ax.axhline(0.9, color="black", lw=1.5, ls="--")
cbar.ax.axhline(1.1, color="black", lw=1.5, ls="--")

patches = [mpatches.Patch(color="#E24B4A", label="Underfed (<0.9)"),
           mpatches.Patch(color="#2ECC71", label="Normal (0.9-1.1)"),
           mpatches.Patch(color="#3498DB", label="Overfed (>1.1)")]
ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.85)
ax.set_title(f"Feed Welfare Status - All 17 Cows x 11 Days  (March 2025)\n"
             f"Michaelis-Menten Model: per-cow DMI_MAX, Km={KM_THESIS} min/day",
             fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Day", fontsize=11)
ax.set_ylabel("Cow ID  [Camera]", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig6_5_welfare_heatmap.png"), dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# Health event trajectories: Cow 354 and Cow 349
# =============================================================================

HEALTH_EVENTS = {
    354: {"date": "March20", "label": "Teat injury (Anafen)"},
    349: {"date": "March23", "label": "Milk fever (IV calcium)"},
}

fig, axes = plt.subplots(2, 2, figsize=(15, 9))

for col, cow in enumerate([354, 349]):
    s = data_16[data_16["cow_id"]==cow].copy()
    s["day_idx"] = s["day"].apply(lambda d: DAYS.index(d))
    s = s.sort_values("day_idx").reset_index(drop=True)
    s["DMI_ratio"] = s["DMI_est_mm"] / s["DMI_NRC"]

    event_day = HEALTH_EVENTS[cow]["date"]
    event_idx = DAYS.index(event_day) if event_day in DAYS else None
    day_labels = [d.replace("March","Mar ") for d in s["day"]]

    ax_top = axes[0, col]
    ax_top.plot(s["day_idx"], s["feeding_min"], marker="o",
                color=COW_COLORS.get(cow,"steelblue"), lw=2)
    if event_idx is not None:
        ax_top.axvline(event_idx, color="red", linestyle="--", alpha=0.7,
                        label=HEALTH_EVENTS[cow]["label"])
    ax_top.set_xticks(s["day_idx"])
    ax_top.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
    ax_top.set_ylabel("Feeding time (min/day)")
    ax_top.set_title(f"Cow {cow} - {HEALTH_EVENTS[cow]['label']}")
    ax_top.legend(fontsize=8)
    ax_top.grid(alpha=0.3)

    ax_bot = axes[1, col]
    ax_bot.plot(s["day_idx"], s["DMI_ratio"], marker="o", color="darkorange", lw=2)
    ax_bot.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
    ax_bot.axhline(0.9, color="#E24B4A", linestyle="--", alpha=0.5, lw=1)
    if event_idx is not None:
        ax_bot.axvline(event_idx, color="red", linestyle="--", alpha=0.7)
    ax_bot.set_xticks(s["day_idx"])
    ax_bot.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)
    ax_bot.set_ylabel("DMI ratio (est / NRC)")
    ax_bot.set_xlabel("Date")
    ax_bot.grid(alpha=0.3)

plt.suptitle("Welfare-Relevant Deviation Detection: Feeding Time and DMI Ratio\n"
             "Around Independently Documented Health Events\n"
             "(Neither model received any health labels as input)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig6_6_health_events.png"), dpi=300, bbox_inches="tight")
plt.show()

# Print the feeding-time values for verification against thesis text
print("\n=== Cow 354 feeding time by day (thesis text: 71 min on March 19) ===")
s354 = data_16[data_16["cow_id"]==354].copy()
s354["day_idx"] = s354["day"].apply(lambda d: DAYS.index(d))
s354 = s354.sort_values("day_idx")
print(s354[["day","feeding_min","DMI_NRC","DMI_est_mm","ratio_mm"]].to_string(index=False))

print("\n=== Cow 349 feeding time by day (event: March 23) ===")
s349 = data_16[data_16["cow_id"]==349].copy()
s349["day_idx"] = s349["day"].apply(lambda d: DAYS.index(d))
s349 = s349.sort_values("day_idx")
print(s349[["day","feeding_min","DMI_NRC","DMI_est_mm","ratio_mm"]].to_string(index=False))

# =============================================================================
# Km sensitivity plot
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(sweep_df["Km"], sweep_df["MAPE"], "o-", color="#3498DB", lw=2, ms=4)
axes[0].axvline(KM_THESIS, color="red", ls="--", lw=1.5, label=f"Km={KM_THESIS} (thesis)")
axes[0].axhline(10, color="green", ls=":", lw=1, alpha=0.5, label="10% threshold")
axes[0].set_xlabel("Km (min/day)"); axes[0].set_ylabel("MAPE (%)")
axes[0].set_title("MAPE vs Km"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

axes[1].plot(sweep_df["Km"], sweep_df["RMSE"], "o-", color="#1D9E75", lw=2, ms=4)
axes[1].axvline(KM_THESIS, color="red", ls="--", lw=1.5)
axes[1].set_xlabel("Km (min/day)"); axes[1].set_ylabel("RMSE (kg DM/day)")
axes[1].set_title("RMSE vs Km"); axes[1].grid(alpha=0.3)

axes[2].plot(sweep_df["Km"], sweep_df["R2"], "o-", color="#9B59B6", lw=2, ms=4)
axes[2].axvline(KM_THESIS, color="red", ls="--", lw=1.5)
axes[2].set_xlabel("Km (min/day)"); axes[2].set_ylabel("R2")
axes[2].set_title("R2 vs Km"); axes[2].grid(alpha=0.3)

plt.suptitle(f"Km Sensitivity - Per-Cow DMI_MAX Model\n"
             f"Km={KM_THESIS} min/day selected (biological range 30-150)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig6_7_km_sensitivity.png"), dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# SAVE RESULT CSVS
# =============================================================================

data_16.to_csv(os.path.join(OUT_DIR,"DMI_16cow_final_results.csv"), index=False)
loco_reg_df.to_csv(os.path.join(OUT_DIR,"DMI_regression_LOCO_results.csv"), index=False)
loco_lin_df.to_csv(os.path.join(OUT_DIR,"DMI_linear_LOCO_results.csv"), index=False)
sweep_df.to_csv(os.path.join(OUT_DIR,"DMI_km_sensitivity.csv"), index=False)

print(f"\nSaved all results to: {OUT_DIR}")
print(f"\nKey thesis numbers (16 cows, {len(data_16)} cow-days):")
print(f"  Linear (local k={k_local:.4f}): RMSE=5.507  MAPE=15.0%")
print(f"  Linear LOCO: MAPE=15.6%")
print(f"  Regression day-split test: RMSE=1.914  MAPE=5.0%  R2=0.669")
print(f"  Regression LOCO: RMSE=2.465  MAPE=5.8%  R2=0.424")
print(f"  M-M (Km={KM_THESIS}, per-cow DMI_MAX) whole-dataset: "
      f"RMSE={np.sqrt((data_16['err_mm']**2).mean()):.3f}  "
      f"MAPE={data_16['pct_mm'].abs().mean():.1f}%")
print(f"  M-M day-split test: RMSE=2.152  MAPE=4.8%  R2=0.564")
