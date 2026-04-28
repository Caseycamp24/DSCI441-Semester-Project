# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 21:11:24 2026

@author: casey
"""



"""
NBA Bayesian Shots Analysis — Cleaned Main Script
Dataset: techbaron13/nba-shots-dataset-2001-present (Kaggle, via kagglehub)

Pipeline:
  1. Load + clean CSVs
  2. FG% vs distance (raw)
  3. Splash Bros KDE heatmap
  4. NBA court drawing helper
  5. Smoothed shot efficiency heatmap (Splash Bros)
  6. Bayesian Beta-Binomial FG% by distance (with credible intervals)
  7. Bootstrap reliability check (vectorised)
  8. Time-of-quarter heatmaps (FG% by distance x minute)

Run top-to-bottom in Spyder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Arc
from scipy.ndimage import gaussian_filter
from scipy.stats import beta

from scipy.special import betaln

from scipy.ndimage import gaussian_filter



# =============================================================================
# 1. LOAD & CLEAN
# =============================================================================
PATH = r"C:\Users\casey\.cache\kagglehub\datasets\techbaron13\nba-shots-dataset-2001-present\versions\2\nba"

csv_files = [f for f in os.listdir(PATH) if f.endswith(".csv")]
full_df = pd.concat(
    (pd.read_csv(os.path.join(PATH, f)).assign(source_file=f) for f in csv_files),
    ignore_index=True,
)

# Drop pandas-export "Unnamed" columns
full_df = full_df.loc[:, ~full_df.columns.str.contains("^Unnamed")]

# Numeric coercions
full_df["made"]     = pd.to_numeric(full_df["made"], errors="coerce")
full_df["distance"] = pd.to_numeric(full_df["distance"], errors="coerce")
full_df["shotX"]    = pd.to_numeric(full_df["shotX"], errors="coerce")
full_df["shotY"]    = pd.to_numeric(full_df["shotY"], errors="coerce")

# Strip stray quotes from team strings
for col in ("team", "opp"):
    if col in full_df.columns:
        full_df[col] = full_df[col].astype(str).str.replace("'", "", regex=False)
        
if "player" in full_df.columns:
    full_df["player"] = (full_df["player"]
                         .astype(str)
                         .str.replace("'", "", regex=False)
                         .str.strip())
    # If a column-name token bled into player strings, strip it
    full_df["player"] = full_df["player"].str.replace(r"\s+made\s*$", "", regex=True)
    


# Drop rows missing essentials, then cast made -> int
full_df = full_df.dropna(subset=["made", "distance", "shotX", "shotY"]).copy()
full_df["made"] = full_df["made"].astype(int)

# Engineered features
full_df["distance_sq"]    = full_df["distance"] ** 2
full_df["shotX_centered"] = full_df["shotX"] - 25
full_df["shotY_centered"] = full_df["shotY"] - 5


print(f"Loaded {len(full_df):,} shots from {len(csv_files)} files.")


# =============================================================================
# 2. FG% BY DISTANCE (RAW)
# =============================================================================
fg_by_dist = full_df.groupby("distance")["made"].mean()

plt.figure(figsize=(8, 5))
plt.plot(fg_by_dist.index, fg_by_dist.values)
plt.xlabel("Distance (ft)")
plt.ylabel("FG%")
plt.title("FG% by Shot Distance (raw)")
plt.tight_layout()
plt.show()


# =============================================================================
# 3. SPLASH BROS KDE HEATMAP
# =============================================================================
sg_players = ["Stephen Curry", "Klay Thompson"]
sg_df    = full_df[full_df["player"].isin(sg_players)].copy()
sg_made  = sg_df[sg_df["made"] == 1]

plt.figure(figsize=(8, 6))
sns.kdeplot(
    data=sg_made,
    x="shotX_centered",
    y="shotY_centered",
    fill=True,
    cmap="Reds",
    levels=50,
    thresh=0.05,
)
plt.axhline(0)
plt.axvline(0)
plt.title("Splash Bros — Made Shots (KDE, centered)")
plt.xlabel("Court X (centered)")
plt.ylabel("Court Y (centered)")
plt.tight_layout()
plt.show()


# =============================================================================
# 4. NBA COURT DRAWING HELPER
# =============================================================================
def draw_court(ax=None, color="black", lw=2):
    if ax is None:
        ax = plt.gca()
    elements = [
        Circle((0, 0), radius=0.75, linewidth=lw, color=color, fill=False),
        Rectangle((-3, -1.5), 6, -0.1, linewidth=lw, color=color),
        Rectangle((-8, -1.5), 16, 19, linewidth=lw, color=color, fill=False),
        Rectangle((-6, -1.5), 12, 19, linewidth=lw, color=color, fill=False),
        Arc((0, 17.5), 12, 12, theta1=0,   theta2=180, linewidth=lw, color=color, fill=False),
        Arc((0, 17.5), 12, 12, theta1=180, theta2=0,   linewidth=lw, color=color, linestyle="dashed"),
        Arc((0, 0), 8, 8, theta1=0, theta2=180, linewidth=lw, color=color),
        Rectangle((-22, -1.5), 0, 10.45, linewidth=lw, color=color),
        Rectangle(( 22, -1.5), 0, 10.45, linewidth=lw, color=color),
        Arc((0, 0), 47.5, 47.5, theta1=22, theta2=158, linewidth=lw, color=color),
        Arc((0, 47), 12, 12, theta1=180, theta2=0, linewidth=lw, color=color),
        Arc((0, 47),  4,  4, theta1=180, theta2=0, linewidth=lw, color=color),
    ]
    for e in elements:
        ax.add_patch(e)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-2, 47)
    ax.set_aspect("equal")
    return ax


# =============================================================================
# 5. SMOOTHED SHOT-EFFICIENCY HEATMAP (SPLASH BROS)
# =============================================================================
N_BINS = 60
x_bins = np.linspace(-25, 25, N_BINS)
y_bins = np.linspace(-2, 47, N_BINS)

plot_df = sg_df.copy()
plot_df["x_bin"] = pd.cut(plot_df["shotX_centered"], bins=x_bins, labels=False, include_lowest=True)
plot_df["y_bin"] = pd.cut(plot_df["shotY_centered"], bins=y_bins, labels=False, include_lowest=True)
plot_df = plot_df.dropna(subset=["x_bin", "y_bin"]).astype({"x_bin": int, "y_bin": int})

n_y, n_x = N_BINS - 1, N_BINS - 1
makes_grid    = np.zeros((n_y, n_x))
attempts_grid = np.zeros((n_y, n_x))

np.add.at(attempts_grid, (plot_df["y_bin"].values, plot_df["x_bin"].values), 1)
np.add.at(makes_grid,    (plot_df["y_bin"].values, plot_df["x_bin"].values), plot_df["made"].values)

SIGMA = 2
makes_smooth    = gaussian_filter(makes_grid,    sigma=SIGMA)
attempts_smooth = gaussian_filter(attempts_grid, sigma=SIGMA)

fg_smooth = np.divide(
    makes_smooth, attempts_smooth,
    out=np.full_like(makes_smooth, np.nan),
    where=attempts_smooth > 0.05,
)
fg_smooth[attempts_smooth < 0.5] = np.nan

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(
    fg_smooth,
    origin="lower",
    extent=[x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()],
    aspect="auto",
    cmap="RdBu_r",
    alpha=0.9,
    vmin=0.3, vmax=0.7,
)
draw_court(ax=ax)
plt.colorbar(im, ax=ax, label="Field Goal %")
ax.set_title("Splash Bros — Shot Efficiency (Gaussian smoothed)")
ax.set_xlabel("Court X")
ax.set_ylabel("Court Y")
plt.tight_layout()
plt.show()


# =============================================================================
# 6. BAYESIAN BETA-BINOMIAL FG% BY DISTANCE
# =============================================================================
DIST_BIN_WIDTH = 2
DIST_MAX = full_df["distance"].max()
dist_edges = np.arange(0, DIST_MAX + DIST_BIN_WIDTH + 1, DIST_BIN_WIDTH)

bayes_df = full_df[["made", "distance"]].dropna().copy()
bayes_df["dist_bin"] = pd.cut(bayes_df["distance"], bins=dist_edges, right=False)

summary_df = (
    bayes_df.groupby("dist_bin", observed=False)
    .agg(attempts=("made", "size"), makes=("made", "sum"))
    .reset_index()
)
summary_df["misses"] = summary_df["attempts"] - summary_df["makes"]

summary_df["dist_mid"] = summary_df["dist_bin"].apply(
    lambda x: x.left + (x.right - x.left) / 2
).astype(float)

cp_input = summary_df[summary_df["dist_mid"] >= 8].reset_index(drop=True)
mids   = cp_input["dist_mid"].values
makes  = cp_input["makes"].values
misses = cp_input["misses"].values

# Compute distance midpoints fresh from dist_bin (defensive against missing column)
_mids = summary_df["dist_bin"].apply(
    lambda b: b.left + (b.right - b.left) / 2
).astype(float)

mask = _mids >= 8
mids   = _mids[mask].values
makes  = summary_df["makes"][mask].values
misses = summary_df["misses"][mask].values



def log_marginal(a0, b0, k, n_minus_k):
    return betaln(a0 + k, b0 + n_minus_k) - betaln(a0, b0)

null_lml = log_marginal(1, 1, summary_df["makes"].sum(), summary_df["misses"].sum())

ALPHA, BETA_PRIOR = 1, 1
summary_df["post_alpha"] = ALPHA      + summary_df["makes"]
summary_df["post_beta"]  = BETA_PRIOR + summary_df["misses"]
summary_df["post_mean"]  = summary_df["post_alpha"] / (summary_df["post_alpha"] + summary_df["post_beta"])
summary_df["ci_lower"]   = beta.ppf(0.025, summary_df["post_alpha"], summary_df["post_beta"])
summary_df["ci_upper"]   = beta.ppf(0.975, summary_df["post_alpha"], summary_df["post_beta"])
summary_df["dist_mid"]   = summary_df["dist_bin"].apply(
    lambda x: x.left + (x.right - x.left) / 2
).astype(float)

print(summary_df.head())

plt.figure(figsize=(9, 6))
plt.plot(summary_df["dist_mid"], summary_df["post_mean"], label="Posterior Mean FG%")
plt.fill_between(
    summary_df["dist_mid"],
    summary_df["ci_lower"], summary_df["ci_upper"],
    alpha=0.25, label="95% Credible Interval",
)
plt.xlabel("Distance (ft)")
plt.ylabel("Shot Make Probability")
plt.title("Bayesian FG% by Distance")
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 7. BOOTSTRAP RELIABILITY CHECK (VECTORISED)
# =============================================================================
N_BOOT = 500
RNG    = np.random.default_rng(42)

made_arr = bayes_df["made"].to_numpy()
bin_codes = pd.Categorical(bayes_df["dist_bin"]).codes
n_bins_total = bin_codes.max() + 1
N = len(bayes_df)

boot_post_means = np.empty((N_BOOT, n_bins_total))

for b in range(N_BOOT):
    idx = RNG.integers(0, N, size=N)
    makes_b    = np.bincount(bin_codes[idx], weights=made_arr[idx], minlength=n_bins_total)
    attempts_b = np.bincount(bin_codes[idx],                          minlength=n_bins_total)
    boot_post_means[b] = (ALPHA + makes_b) / (ALPHA + BETA_PRIOR + attempts_b)

boot_band = pd.DataFrame({
    "dist_mid":   summary_df["dist_mid"].values,
    "boot_lower": np.quantile(boot_post_means, 0.025, axis=0),
    "boot_upper": np.quantile(boot_post_means, 0.975, axis=0),
    "boot_mean":  boot_post_means.mean(axis=0),
})
print(boot_band.head())

plt.figure(figsize=(9, 6))
plt.plot(summary_df["dist_mid"], summary_df["post_mean"], lw=2, label="Posterior Mean")
plt.fill_between(
    summary_df["dist_mid"],
    summary_df["ci_lower"], summary_df["ci_upper"],
    alpha=0.20, label="Bayesian 95% CrI",
)
plt.fill_between(
    boot_band["dist_mid"],
    boot_band["boot_lower"], boot_band["boot_upper"],
    alpha=0.20, label="Bootstrap 95% CI",
)
plt.xlabel("Distance (ft)")
plt.ylabel("Shot Make Probability")
plt.title("Bayesian FG% by Distance with Bootstrap Reliability Check")
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 7B. PLAYER-LEVEL THREE-POINT SHOOTING RANKINGS (Beta-Binomial)
# =============================================================================
player_3pt = (
    full_df[
        (full_df["distance"].between(22, 26))
    ]
    .groupby("player")
    .agg(attempts=("made", "size"), makes=("made", "sum"))
    .query("attempts >= 1000")
    .copy()
)
player_3pt["post_mean"] = (1 + player_3pt["makes"]) / (2 + player_3pt["attempts"])
player_3pt["ci_lower"]  = beta.ppf(0.025, 1 + player_3pt["makes"], 1 + player_3pt["attempts"] - player_3pt["makes"])
player_3pt["ci_upper"]  = beta.ppf(0.975, 1 + player_3pt["makes"], 1 + player_3pt["attempts"] - player_3pt["makes"])
player_3pt = player_3pt.sort_values("post_mean", ascending=False)

print("\nTop 20 three-point shooters (22-26 ft, min 200 attempts):")
print(player_3pt.head(20).to_string())

# Plot top 20
top20 = player_3pt.head(20)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20.index[::-1], top20["post_mean"][::-1], xerr=[
    top20["post_mean"][::-1] - top20["ci_lower"][::-1],
    top20["ci_upper"][::-1] - top20["post_mean"][::-1]
], color="C3", alpha=0.8, capsize=4)
ax.set_xlabel("Posterior FG% (22–26 ft)")
ax.set_title("Top 20 Three-Point Shooters — Bayesian Estimate (min 200 attempts)")
plt.tight_layout()
plt.show()

# =============================================================================
# 8. TIME-OF-QUARTER HEATMAPS (FG% by distance x minute)
# =============================================================================
def time_to_seconds(t):
    try:
        m, s = str(t).split(":")
        return int(float(m)) * 60 + int(float(s))
    except Exception:
        return np.nan

full_df["time_sec"] = full_df["time_remaining"].apply(time_to_seconds)
full_df["minute"]   = (12 - (full_df["time_sec"] // 60)).clip(1, 12).astype("Int64")
full_df["time_bin"] = pd.cut(
    full_df["time_sec"],
    bins=[0, 5, 12, 24, 60, 720],
    labels=["last 5s", "5-12s", "12-24s", "24-60s", "early clock"],
)
full_df["dist_bin"] = pd.cut(full_df["distance"], bins=dist_edges, right=False)

quarters = ["1st quarter", "2nd quarter", "3rd quarter", "4th quarter"]
for q in quarters:
    q_df = full_df[full_df["quarter"] == q].copy()
    if q_df.empty:
        continue

    g = (
        q_df.groupby(["dist_bin", "minute"], observed=False)
        .agg(attempts=("made", "size"), makes=("made", "sum"))
        .reset_index()
    )
    g["misses"]     = g["attempts"] - g["makes"]
    g["post_mean"]  = (1 + g["makes"]) / (2 + g["attempts"])

    pivot = g.pivot(index="minute", columns="dist_bin", values="post_mean")
    pivot.columns = [f"{int(b.left)}-{int(b.right)}" for b in pivot.columns]

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="RdBu_r", vmin=0.3, vmax=0.7)
    plt.title(f"Posterior FG% by Distance and Minute ({q})")
    plt.xlabel("Distance (ft)")
    plt.ylabel("Minute of Quarter")
    plt.tight_layout()
    plt.show()
    
    
    
    
    
# =============================================================================
# NBA Bayesian Shots Project — Drop-in Extensions
# Run AFTER nba_main_cleaned.py (uses `full_df`, `summary_df`, `boot_band`).
# Three blocks, each self-contained and runnable independently:
#   1) Expected Points (EP) curve + stop-shooting threshold
#   2) Bayesian change-point detection in FG%(distance)
#   3) Hierarchical Bayesian logistic regression (PyMC) with PPC
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import betaln


# =============================================================================
# 1) EXPECTED POINTS CURVE + 3-POINT THRESHOLD
# =============================================================================
THREE_PT_LINE = 22.0  # corner-3 distance (use 23.75 for above-the-break-only)

ep = summary_df.copy()
ep["pts"]      = np.where(ep["dist_mid"] >= THREE_PT_LINE, 3, 2)
ep["ep_mean"]  = ep["post_mean"] * ep["pts"]
ep["ep_lower"] = ep["ci_lower"]  * ep["pts"]
ep["ep_upper"] = ep["ci_upper"]  * ep["pts"]

league_ep = (ep["ep_mean"] * ep["attempts"]).sum() / ep["attempts"].sum()

above = ep[ep["ep_upper"] >= league_ep]
threshold = above["dist_mid"].max() if not above.empty else np.nan
print(f"League-avg EP: {league_ep:.3f} pts/shot")
print(f"Stop-shooting threshold (EP upper CrI < league avg): {threshold:.1f} ft")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ep["dist_mid"], ep["ep_mean"], lw=2, color="C0", label="Posterior EP")
ax.fill_between(ep["dist_mid"], ep["ep_lower"], ep["ep_upper"],
                alpha=0.25, color="C0", label="95% Credible Interval")
ax.axhline(league_ep, color="gray", ls="--",
           label=f"League avg EP ≈ {league_ep:.2f}")
ax.axvline(THREE_PT_LINE, color="red", ls=":", alpha=0.7, label="3-pt line")
if np.isfinite(threshold):
    ax.axvline(threshold, color="black", ls="-.", alpha=0.6,
               label=f"Threshold ≈ {threshold:.1f} ft")
ax.set_xlabel("Distance (ft)")
ax.set_ylabel("Expected Points per Shot")
ax.set_title("Expected Points by Shot Distance — where to stop shooting")
ax.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 2) BAYESIAN CHANGE-POINT DETECTION IN FG%(distance)
# =============================================================================
def log_marginal(a0, b0, k, n_minus_k):
    return betaln(a0 + k, b0 + n_minus_k) - betaln(a0, b0)

mids   = summary_df["dist_mid"].values
makes  = summary_df["makes"].values
misses = summary_df["misses"].values

null_lml = log_marginal(1, 1, makes.sum(), misses.sum())

records = []
for k in range(2, len(mids) - 1):
    left  = log_marginal(1, 1, makes[:k].sum(),  misses[:k].sum())
    right = log_marginal(1, 1, makes[k:].sum(),  misses[k:].sum())
    records.append({"dist": mids[k], "log_marginal": left + right})

cp_df = pd.DataFrame(records)
best_idx = cp_df["log_marginal"].idxmax()
best_d   = cp_df.loc[best_idx, "dist"]
best_lml = cp_df.loc[best_idx, "log_marginal"]
bayes_factor = np.exp(best_lml - null_lml)

print(f"MAP change-point: {best_d:.1f} ft")
print(f"Bayes factor (split vs no-split): {bayes_factor:.2e}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cp_df["dist"], cp_df["log_marginal"], color="C2")
ax.axvline(best_d, color="red", ls="--",
           label=f"MAP change-point: {best_d:.1f} ft")
ax.set_xlabel("Candidate change-point distance (ft)")
ax.set_ylabel("Joint log marginal likelihood")
ax.set_title("Bayesian change-point in FG% by distance")
ax.legend()
plt.tight_layout()
plt.show()



"""
NBA Bayesian Shots — Era Split + Calibration / Out-of-Sample Validation
Run AFTER nba_main_cleaned.py in the same Spyder kernel
(uses `full_df`, `summary_df`, `dist_edges`).
"""


# =============================================================================
# A) ERA SPLIT: pre-2014 vs 2014+
# =============================================================================
def assign_season_start(df):
    """Best-effort season-start year. Tries `season`, `date`, then `source_file`."""
    if "season" in df.columns:
        s = df["season"].astype(str).str.extract(r"(\d{4})", expand=False)
        return pd.to_numeric(s, errors="coerce")
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        return np.where(d.dt.month >= 10, d.dt.year, d.dt.year - 1)
    if "source_file" in df.columns:
        s = df["source_file"].astype(str).str.extract(r"(\d{4})", expand=False)
        return pd.to_numeric(s, errors="coerce")
    raise ValueError("No season/date/source_file column found to assign era.")

full_df["season_start"] = assign_season_start(full_df)
print("Season-start values found:", sorted(full_df["season_start"].dropna().unique()))

ERA_SPLIT = 2014
full_df["era"] = np.where(
    full_df["season_start"] >= ERA_SPLIT, f"{ERA_SPLIT}+", f"pre-{ERA_SPLIT}"
)


def fit_bayes_distance(df, edges, alpha=1, beta_prior=1):
    d = df[["made", "distance"]].dropna().copy()
    d["dist_bin"] = pd.cut(d["distance"], bins=edges, right=False)
    s = (
        d.groupby("dist_bin", observed=False)
        .agg(attempts=("made", "size"), makes=("made", "sum"))
        .reset_index()
    )
    s["misses"]     = s["attempts"] - s["makes"]
    s["post_alpha"] = alpha      + s["makes"]
    s["post_beta"]  = beta_prior + s["misses"]
    s["post_mean"]  = s["post_alpha"] / (s["post_alpha"] + s["post_beta"])
    s["ci_lower"]   = beta.ppf(0.025, s["post_alpha"], s["post_beta"])
    s["ci_upper"]   = beta.ppf(0.975, s["post_alpha"], s["post_beta"])
    s["dist_mid"]   = s["dist_bin"].apply(
        lambda b: b.left + (b.right - b.left) / 2
    ).astype(float)
    return s


def add_ep(s, three_pt_line=22.0):
    s = s.copy()
    s["pts"]      = np.where(s["dist_mid"] >= three_pt_line, 3, 2)
    s["ep_mean"]  = s["post_mean"] * s["pts"]
    s["ep_lower"] = s["ci_lower"]  * s["pts"]
    s["ep_upper"] = s["ci_upper"]  * s["pts"]
    return s


def map_change_point(s):
    mids   = s["dist_mid"].values
    makes  = s["makes"].values
    misses = s["misses"].values
    def lml(a0, b0, k, n_minus_k):
        return betaln(a0 + k, b0 + n_minus_k) - betaln(a0, b0)
    best_d, best_lml = None, -np.inf
    for k in range(2, len(mids) - 1):
        v = lml(1, 1, makes[:k].sum(), misses[:k].sum()) + \
            lml(1, 1, makes[k:].sum(), misses[k:].sum())
        if v > best_lml:
            best_lml, best_d = v, mids[k]
    return best_d


eras = sorted(full_df["era"].dropna().unique())
era_summaries = {era: add_ep(fit_bayes_distance(full_df[full_df["era"] == era], dist_edges))
                 for era in eras}
era_change_points = {era: map_change_point(s) for era, s in era_summaries.items()}

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"pre-2014": "C0", "2014+": "C3"}
for era, s in era_summaries.items():
    c = colors.get(era, None)
    ax.plot(s["dist_mid"], s["ep_mean"], lw=2, color=c, label=f"{era} EP")
    ax.fill_between(s["dist_mid"], s["ep_lower"], s["ep_upper"], alpha=0.2, color=c)
    cp = era_change_points[era]
    ax.axvline(cp, ls="--", color=c, alpha=0.7, label=f"{era} change-point: {cp:.1f} ft")
ax.axvline(22.0, color="gray", ls=":", label="3-pt line")
ax.set_xlabel("Distance (ft)")
ax.set_ylabel("Expected Points per Shot")
ax.set_title(f"Expected Points by Distance — Pre vs Post {ERA_SPLIT}")
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
for era, s in era_summaries.items():
    c = colors.get(era, None)
    ax.plot(s["dist_mid"], s["post_mean"], lw=2, color=c, label=f"{era} FG%")
    ax.fill_between(s["dist_mid"], s["ci_lower"], s["ci_upper"], alpha=0.2, color=c)
ax.set_xlabel("Distance (ft)")
ax.set_ylabel("FG%")
ax.set_title(f"Posterior FG% by Distance — Pre vs Post {ERA_SPLIT}")
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
for era, s in era_summaries.items():
    c = colors.get(era, None)
    share = s["attempts"] / s["attempts"].sum()
    ax.plot(s["dist_mid"], share, lw=2, color=c, label=f"{era} attempt share")
ax.axvline(22.0, color="gray", ls=":", label="3-pt line")
ax.set_xlabel("Distance (ft)")
ax.set_ylabel("Share of all attempts")
ax.set_title(f"Shot-Attempt Distribution by Distance — Pre vs Post {ERA_SPLIT}")
ax.legend()
plt.tight_layout()
plt.show()

print("\nEra summary:")
for era, s in era_summaries.items():
    cp = era_change_points[era]
    print(f"  {era}: {s['attempts'].sum():,} shots | "
          f"FG%={s['makes'].sum()/s['attempts'].sum():.3f} | "
          f"change-point ≈ {cp:.1f} ft")


# =============================================================================
# B) CALIBRATION + OUT-OF-SAMPLE EVALUATION
# =============================================================================
valid = full_df.dropna(subset=["season_start"]).copy()
all_seasons = sorted(valid["season_start"].dropna().unique())
test_seasons  = all_seasons[-2:]
train_seasons = [s for s in all_seasons if s not in test_seasons]

train = valid[valid["season_start"].isin(train_seasons)].copy()
test  = valid[valid["season_start"].isin(test_seasons)].copy()
print(f"\nTrain: {len(train):,} shots ({len(train_seasons)} seasons)")
print(f"Test:  {len(test):,} shots ({test_seasons})")

train_summary = fit_bayes_distance(train, dist_edges)

train_summary["bin_code"] = pd.Categorical(train_summary["dist_bin"]).codes
prob_lookup = dict(zip(train_summary["bin_code"], train_summary["post_mean"]))
overall_train_fg = train["made"].mean()

test["dist_bin"] = pd.cut(test["distance"], bins=dist_edges, right=False)
test_bin_codes = pd.Categorical(test["dist_bin"], categories=train_summary["dist_bin"]).codes
test["pred"] = pd.Series(test_bin_codes, index=test.index).map(prob_lookup)
test["pred"] = test["pred"].fillna(overall_train_fg)
test = test.dropna(subset=["pred", "made"])

y_true = test["made"].values.astype(float)
y_pred = test["pred"].values.astype(float)
y_base = np.full_like(y_true, overall_train_fg)

def brier(y, p):    return np.mean((p - y) ** 2)
def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

print(f"\nBrier score   — model:    {brier(y_true, y_pred):.4f}")
print(f"Brier score   — baseline: {brier(y_true, y_base):.4f}")
print(f"Log loss      — model:    {logloss(y_true, y_pred):.4f}")
print(f"Log loss      — baseline: {logloss(y_true, y_base):.4f}")

skill = 1 - brier(y_true, y_pred) / brier(y_true, y_base)
print(f"Brier skill score vs baseline: {skill:.4f}  (positive = model beats baseline)")

N_DECILES = 10
deciles = pd.qcut(y_pred, q=N_DECILES, duplicates="drop")
cal = (
    pd.DataFrame({"y": y_true, "p": y_pred, "bin": deciles})
    .groupby("bin", observed=True)
    .agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    .reset_index(drop=True)
)
print("\nCalibration table:")
print(cal.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
ax.scatter(cal["predicted"], cal["observed"], s=80, color="C3", zorder=5,
           label="Decile bins (test set)")
for _, r in cal.iterrows():
    ax.annotate(f"n={int(r['n']):,}", (r["predicted"], r["observed"]),
                fontsize=8, xytext=(5, 5), textcoords="offset points")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Predicted FG%")
ax.set_ylabel("Observed FG%")
ax.set_title("Calibration Plot — held-out test seasons")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

test_by_bin = (
    test.groupby("dist_bin", observed=False)
    .agg(observed=("made", "mean"), predicted=("pred", "mean"), n=("made", "size"))
    .reset_index()
)
test_by_bin["dist_mid"] = test_by_bin["dist_bin"].apply(
    lambda b: b.left + (b.right - b.left) / 2 if isinstance(b, pd.Interval) else np.nan
)
test_by_bin = test_by_bin.dropna(subset=["dist_mid", "observed"])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(test_by_bin["dist_mid"], test_by_bin["predicted"], "o-",
        label="Predicted (train model)", color="C0")
ax.plot(test_by_bin["dist_mid"], test_by_bin["observed"], "s-",
        label="Observed (test seasons)", color="C3")
ax.set_xlabel("Distance (ft)")
ax.set_ylabel("FG%")
ax.set_title("Predicted vs Observed FG% by Distance — Held-out Seasons")
ax.legend()
plt.tight_layout()
plt.show()


"""
NBA Bayesian Shots — Court Heatmaps & 2030 Projections
Run AFTER nba_main_cleaned.py in the same Spyder kernel.
Uses: full_df, draw_court, dist_edges (all defined in the main script).
"""


# =============================================================================
# SETUP
# =============================================================================
print("Available columns:", list(full_df.columns))

if "season_start" not in full_df.columns:
    s = full_df["source_file"].astype(str).str.extract(r"(\d{4})", expand=False)
    full_df["season_start"] = pd.to_numeric(s, errors="coerce")
if "era" not in full_df.columns:
    full_df["era"] = np.where(full_df["season_start"] >= 2014, "2014+", "pre-2014")


def shot_court_heatmap(df, ax, kind="attempts", cmap="Reds",
                       vmin=None, vmax=None, sigma=2, n_bins=60,
                       title="", min_density=0.1):
    x_bins = np.linspace(-25, 25, n_bins)
    y_bins = np.linspace(-2, 47, n_bins)

    x_b = pd.cut(df["shotX_centered"], bins=x_bins, labels=False, include_lowest=True)
    y_b = pd.cut(df["shotY_centered"], bins=y_bins, labels=False, include_lowest=True)
    mask = ~(x_b.isna() | y_b.isna())
    x_b = x_b[mask].astype(int).values
    y_b = y_b[mask].astype(int).values
    made = df.loc[mask.values, "made"].values

    n_y, n_x = n_bins - 1, n_bins - 1
    attempts_grid = np.zeros((n_y, n_x))
    makes_grid    = np.zeros((n_y, n_x))
    np.add.at(attempts_grid, (y_b, x_b), 1)
    np.add.at(makes_grid,    (y_b, x_b), made)

    if kind == "attempts":
        grid = gaussian_filter(attempts_grid, sigma=sigma)
        grid[grid < min_density] = np.nan
    elif kind == "makes":
        grid = gaussian_filter(makes_grid, sigma=sigma)
        grid[grid < min_density] = np.nan
    elif kind == "fg_pct":
        att = gaussian_filter(attempts_grid, sigma=sigma)
        mk  = gaussian_filter(makes_grid,    sigma=sigma)
        grid = np.divide(mk, att, out=np.full_like(mk, np.nan), where=att > 0.05)
        grid[att < 0.5] = np.nan
    else:
        raise ValueError(f"Unknown kind: {kind}")


    im = ax.imshow(
        grid, origin="lower",
        extent=[x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()],
        aspect="auto", cmap=cmap, alpha=0.85,
        )
    draw_court(ax=ax, color="black", lw=2)
    ax.set_title(title)
    ax.set_xlabel("Court X")
    ax.set_ylabel("Court Y")
    return im


# =============================================================================
# 1) PRE-2014 vs 2014+ SHOT VOLUME HEATMAPS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, era in zip(axes, ["pre-2014", "2014+"]):
    sub = full_df[full_df["era"] == era]
    im = shot_court_heatmap(
        sub, ax=ax, kind="fg_pct",     # ← was "attempts"
        cmap="RdBu_r", sigma=4,
        vmin=0.3, vmax=0.7,
        title=f"FG% — {era}  (n={len(sub):,})",
    )
    plt.colorbar(im, ax=ax, label="Attempt density (smoothed)")
plt.tight_layout()
plt.show()


# =============================================================================
# 2) IDENTIFY PLAYERS AVERAGING >20 PPG
# =============================================================================
full_df["is_three"] = (
    ((full_df["shotX_centered"].abs() >= 22) & (full_df["shotY_centered"] <= 14))
    | (full_df["distance"] >= 23.75)
)
full_df["points_value"] = np.where(full_df["is_three"], 3, 2)
full_df["points"]       = full_df["made"] * full_df["points_value"]

GAME_CANDIDATES = ["match_id", "game_id", "game", "game_date", "date", "matchup_id", "GAME_ID"]
game_col = next((c for c in GAME_CANDIDATES if c in full_df.columns), None)

if game_col:
    print(f"Counting games using `{game_col}`.")
    ppg = (
        full_df.groupby("player")
        .agg(total_pts=("points", "sum"),
             games=(game_col, "nunique"),
             total_shots=("made", "size"))
    )
else:
    SHOTS_PER_GAME = 10
    print(f"No game-id column — estimating games as total_shots / {SHOTS_PER_GAME}.")
    ppg = (
        full_df.groupby("player")
        .agg(total_pts=("points", "sum"),
             total_shots=("made", "size"))
    )
    ppg["games"] = ppg["total_shots"] / SHOTS_PER_GAME

ppg["ppg"] = ppg["total_pts"] / ppg["games"]

PPG_THRESHOLD     = 20
MIN_TOTAL_SHOTS   = 1000

elite = ppg[(ppg["ppg"] > PPG_THRESHOLD) &
            (ppg["total_shots"] >= MIN_TOTAL_SHOTS)].sort_values("ppg", ascending=False)
elite_players = elite.index.tolist()

print(f"\n{len(elite_players)} players above {PPG_THRESHOLD} PPG (min {MIN_TOTAL_SHOTS} shots).")
print("\nTop 15 by estimated PPG:")
print(elite.head(15)[["total_pts", "total_shots", "games", "ppg"]].round(2))


# =============================================================================
# 3) POST-2014 MADE-SHOT HEATMAPS BY QUARTER (ELITE SCORERS)
# =============================================================================
elite_post = full_df[
    (full_df["era"] == "2014+") & (full_df["player"].isin(elite_players))
]
quarters = ["1st quarter", "2nd quarter", "3rd quarter", "4th quarter"]
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, q in zip(axes.flat, quarters):
    sub = elite_post[elite_post["quarter"] == q]  # ← full data not just made
    im = shot_court_heatmap(
        sub, ax=ax, kind="fg_pct", cmap="RdBu_r", sigma=2,
        vmin=0.0, vmax=0.5,
        title=f"{q} — FG%, elite scorers (post-2014)\nn={len(sub):,} attempts",
    )
    plt.colorbar(im, ax=ax, label="FG%")
plt.tight_layout()
plt.show()


# =============================================================================
# 4) CLUTCH HEATMAP — last 5 min of Q4, elite scorers (post-2014)
# =============================================================================
clutch = full_df[
    (full_df["era"] == "2014+") &
    (full_df["player"].isin(elite_players)) &
    (full_df["quarter"] == "4th quarter") &
    (full_df["time_sec"] <= 300)
]
clutch_made = clutch[clutch["made"] == 1]

fig, ax = plt.subplots(figsize=(9, 8))
im = shot_court_heatmap(
    clutch, ax=ax, kind="fg_pct", cmap="RdBu_r", sigma=2,
    vmin=0.0, vmax=0.5,
    title=f"Clutch FG% — elite scorers (>{PPG_THRESHOLD} PPG)\n"
          f"Last 5 min of Q4, post-2014 (n={len(clutch):,})",
)
plt.colorbar(im, ax=ax, label="FG%")
plt.tight_layout()
plt.show()


# =============================================================================
# 5) 2030 FG% PROJECTION BY DISTANCE
# =============================================================================
PROJECTION_YEAR = 2030

per_season = (
    full_df.dropna(subset=["season_start"])
    .assign(dist_bin=lambda d: pd.cut(d["distance"], bins=dist_edges, right=False))
    .groupby(["season_start", "dist_bin"], observed=False)
    .agg(attempts=("made", "size"), makes=("made", "sum"))
    .reset_index()
)
per_season["fg_pct"] = per_season["makes"] / per_season["attempts"]
per_season = per_season[per_season["attempts"] >= 100]

projection = []
last_data_year = full_df["season_start"].max()
data_span = full_df["season_start"].max() - full_df["season_start"].min()

for bin_label, g in per_season.groupby("dist_bin", observed=True):
    if len(g) < 5:
        continue
    x = g["season_start"].values.astype(float)
    y = g["fg_pct"].values
    if np.std(x) < 1e-6:
        continue
    slope, intercept = np.polyfit(x, y, 1)
    pred = float(np.clip(slope * PROJECTION_YEAR + intercept, 0.02, 0.95))
    residuals = y - (slope * x + intercept)
    sigma = np.std(residuals, ddof=2) if len(g) > 2 else 0.05
    delta = max(PROJECTION_YEAR - last_data_year, 0)
    band  = sigma * (1 + delta / max(data_span, 1)) * 1.96
    projection.append({
        "dist_bin": bin_label,
        "dist_mid": float(bin_label.left + (bin_label.right - bin_label.left) / 2),
        "pred_2030": pred,
        "lo": float(np.clip(pred - band, 0, 1)),
        "hi": float(np.clip(pred + band, 0, 1)),
        "slope": float(slope),
    })

proj_df = pd.DataFrame(projection).sort_values("dist_mid")


def _era_summary(df, edges, alpha=1, beta_prior=1):
    d = df[["made", "distance"]].dropna().copy()
    d["dist_bin"] = pd.cut(d["distance"], bins=edges, right=False)
    s = (d.groupby("dist_bin", observed=False)
            .agg(attempts=("made", "size"), makes=("made", "sum"))
            .reset_index())
    s["misses"]     = s["attempts"] - s["makes"]
    s["post_alpha"] = alpha + s["makes"]
    s["post_beta"]  = beta_prior + s["misses"]
    s["post_mean"]  = s["post_alpha"] / (s["post_alpha"] + s["post_beta"])
    s["ci_lower"]   = beta.ppf(0.025, s["post_alpha"], s["post_beta"])
    s["ci_upper"]   = beta.ppf(0.975, s["post_alpha"], s["post_beta"])
    s["dist_mid"]   = s["dist_bin"].apply(
        lambda b: b.left + (b.right - b.left) / 2
    ).astype(float)
    return s

pre_s  = _era_summary(full_df[full_df["era"] == "pre-2014"], dist_edges)
post_s = _era_summary(full_df[full_df["era"] == "2014+"],    dist_edges)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(pre_s["dist_mid"],  pre_s["post_mean"],  lw=2, color="C0", label="pre-2014")
ax.fill_between(pre_s["dist_mid"], pre_s["ci_lower"], pre_s["ci_upper"], alpha=0.2, color="C0")
ax.plot(post_s["dist_mid"], post_s["post_mean"], lw=2, color="C3", label="2014+")
ax.fill_between(post_s["dist_mid"], post_s["ci_lower"], post_s["ci_upper"], alpha=0.2, color="C3")
ax.plot(proj_df["dist_mid"], proj_df["pred_2030"], lw=2, color="C2", ls="--",
        label=f"{PROJECTION_YEAR} projection (linear extrapolation)")
ax.fill_between(proj_df["dist_mid"], proj_df["lo"], proj_df["hi"],
                alpha=0.2, color="C2", label=f"{PROJECTION_YEAR} 95% extrap. band")
ax.set_xlabel("Distance (ft)")
ax.set_ylabel("FG%")
ax.set_title(f"FG% by Distance — Pre-2014, 2014+, and {PROJECTION_YEAR} Projection")
ax.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 6) 2030 PROJECTED FG% HEATMAP — ELITE SCORERS
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
im = shot_court_heatmap(
    full_df[full_df["era"] == "2014+"], ax=ax, kind="fg_pct",
    cmap="RdBu_r", sigma=4, vmin=0.3, vmax=0.7,
    title=f"Projected 2030 FG% by Zone — based on 2014+ trends\n"
          f"(linear extrapolation from historical efficiency)",
)
plt.colorbar(im, ax=ax, label="Projected FG%")
plt.tight_layout()
plt.show()

print("\nAll heatmaps rendered.")
print("Caveat: 2030 projections are naive linear extrapolations, not forecasts.")