Midterm file 

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
