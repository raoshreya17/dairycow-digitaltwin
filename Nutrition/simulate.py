"""
14-day closed-loop simulation of the proportional feedback controller
across five representative lactation scenarios.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

DMI_MAX  = 28.0   # kg DM/day
K_M      = 62.0   # min/day
K_GAIN   = 0.5
DEADBAND = 0.5    # kg DM/day
U_MIN    = 15.0   # kg DM/day
U_MAX    = 28.0   # kg DM/day
U_INIT   = 18.0   # starting allocation for all scenarios
N_DAYS   = 14

SIGMA_FEED = 12.0
SIGMA_MILK = 0.40
SIGMA_BW   = 0.50

# =============================================================================
# CORE EQUATIONS
# =============================================================================

def mm_estimate(f_feed, dmi_max=DMI_MAX, k_m=K_M):
    return dmi_max * f_feed / (k_m + f_feed)

def mm_inverse(dmi_est, dmi_max=DMI_MAX, k_m=K_M):
    dmi_est = min(dmi_est, dmi_max * 0.999)
    return k_m * dmi_est / (dmi_max - dmi_est)

def ecm(y_milk, fat_kg, protein_kg):
    return 0.327 * y_milk + 12.96 * fat_kg + 7.2 * protein_kg

def nrc_target(bw, dim, y_milk, fat_kg, protein_kg):
    ecm_val = ecm(y_milk, fat_kg, protein_kg)
    return 0.372 * ecm_val + 0.0968 * (bw ** 0.75) * np.exp(-0.192 * (dim / 30 + 3.67))

def controller_step(u_current, dmi_nrc, dmi_est):
    e = dmi_nrc - dmi_est
    adjustment = K_GAIN * e if abs(e) > DEADBAND else 0.0
    return float(np.clip(u_current + adjustment, U_MIN, U_MAX))

# =============================================================================
# SCENARIO DEFINITIONS 
# =============================================================================

SCENARIOS = [
    dict(label="Cow 1  Fresh (DIM 30, 30 kg/day)",
         cow_id="Cow1", bw=560, dim=30,  y_milk=30.0,
         fat_kg=1.14, protein_kg=0.96, f_feed_init=150.0),
    dict(label="Cow 2  Early (DIM 60, 26 kg/day)",
         cow_id="Cow2", bw=590, dim=60,  y_milk=26.0,
         fat_kg=0.99, protein_kg=0.83, f_feed_init=155.0),
    dict(label="Cow 3  Mid (DIM 120, 22 kg/day)",
         cow_id="Cow3", bw=610, dim=120, y_milk=22.0,
         fat_kg=0.84, protein_kg=0.70, f_feed_init=165.0),
    dict(label="Cow 4  Mid-late (DIM 150, 20 kg/day)",
         cow_id="Cow4", bw=630, dim=150, y_milk=20.0,
         fat_kg=0.76, protein_kg=0.64, f_feed_init=175.0),
    dict(label="Cow 5  Late (DIM 200, 18 kg/day)",
         cow_id="Cow5", bw=650, dim=200, y_milk=18.0,
         fat_kg=0.68, protein_kg=0.58, f_feed_init=178.0),
]

# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation(scenarios, n_days=N_DAYS, seed=0):
    rng     = np.random.default_rng(seed)
    results = []

    for sc in scenarios:
        bw         = sc["bw"]
        dim        = sc["dim"]
        y_milk     = sc["y_milk"]
        fat_kg     = sc["fat_kg"]
        protein_kg = sc["protein_kg"]
        f_feed     = sc["f_feed_init"]
        u          = U_INIT

        deviations  = []
        allocations = []
        recovery_day = None

        for day in range(n_days):
            f_noisy    = max(0.0, f_feed + rng.normal(0, SIGMA_FEED))
            bw_noisy   = max(300.0, bw + rng.normal(0, SIGMA_BW))
            milk_noisy = max(0.0, y_milk + rng.normal(0, SIGMA_MILK))

            dmi_est = mm_estimate(f_noisy)
            dmi_nrc = nrc_target(bw_noisy, dim, milk_noisy, fat_kg, protein_kg)
            e       = dmi_nrc - dmi_est

            deviations.append(round(e, 3))
            allocations.append(round(u, 3))

            if recovery_day is None and abs(e) <= DEADBAND:
                recovery_day = day + 1

            u_next = controller_step(u, dmi_nrc, dmi_est)

            # State transition
            target_f = mm_inverse(u_next)
            f_feed   = 0.6 * f_feed + 0.4 * target_f
            u        = u_next
            dim     += 1

        results.append({
            "label":        sc["label"],
            "cow_id":       sc["cow_id"],
            "deviations":   deviations,
            "allocations":  allocations,
            "recovery_day": recovery_day if recovery_day else n_days,
            "e0":           deviations[0],
            "e_final":      deviations[-1],
        })

    return results

# =============================================================================
# TABLE OUTPUT
# =============================================================================

def print_table(results):
    print("\nTable 6.11  Closed-loop simulation (K=0.5, delta=0.5 kg DM/day)")
    print("=" * 90)
    print(f"  {'Scenario':<40}  {'e(0)':>7}  {'e(14)':>7}  {'Recovery':>9}  "
          f"{'Active':>7}  {'Stable':>7}")
    print("-" * 90)
    for r in results:
        active = sum(1 for e in r["deviations"] if abs(e) > DEADBAND)
        stable = N_DAYS - active
        print(f"  {r['label']:<40}  {r['e0']:>+7.2f}  {r['e_final']:>+7.2f}  "
              f"{r['recovery_day']:>8}d  {active:>7}  {stable:>7}")
    print("=" * 90)
    print("  e(0) and e(14) in kg DM/day.  Active = days controller issued adjustment.\n")

# =============================================================================
# FIGURES
# =============================================================================

COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def plot_figures(results, out_dir="."):
    days = list(range(1, N_DAYS + 1))
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax = axes[0]
    ax.axhspan(-DEADBAND, DEADBAND, color="lightgrey", alpha=0.5,
               label=f"Dead-band ({DEADBAND} kg DM/day)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    for i, r in enumerate(results):
        ax.plot(days, r["deviations"], marker="o", markersize=3,
                color=COLOURS[i], label=r["label"])
    ax.set_ylabel("Intake deviation e(t)\n(kg DM/day)", fontsize=11)
    ax.set_title("(a) Deviation convergence", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, lw=0.4, alpha=0.5)

    ax = axes[1]
    for i, r in enumerate(results):
        ax.plot(days, r["allocations"], marker="s", markersize=3,
                color=COLOURS[i], label=r["label"])
    ax.axhline(U_MAX, color="grey", lw=0.8, ls=":", label=f"u_max = {U_MAX} kg/day")
    ax.axhline(U_MIN, color="grey", lw=0.8, ls="-.", label=f"u_min = {U_MIN} kg/day")
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Feed allocation u(t)\n(kg DM/day)", fontsize=11)
    ax.set_title("(b) Feed allocation trajectories", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(14, 30)
    ax.grid(True, lw=0.4, alpha=0.5)

    fig.suptitle(
        "Closed-loop proportional controller simulation\n"
        f"K={K_GAIN}, delta={DEADBAND} kg DM/day, u in [{U_MIN}, {U_MAX}] kg DM/day",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = f"{out_dir}/figure_controller_simulation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out_path}")
    plt.close(fig)

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="14-day controller simulation.")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure.")
    parser.add_argument("--seed",    type=int, default=0, help="Random seed.")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory.")
    args = parser.parse_args()

    print(f"Running 14-day simulation (seed={args.seed}) ...")
    results = run_simulation(SCENARIOS, seed=args.seed)
    print_table(results)

    if not args.no_plot:
        plot_figures(results, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
