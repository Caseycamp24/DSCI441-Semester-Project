"""
NBA Bayesian Shots Analysis -- Streamlit App
Dataset: techbaron13/nba-shots-dataset-2001-present (Kaggle)
 
Run locally:
    streamlit run app.py
 
Deploy on Streamlit Community Cloud:
    Add KAGGLE_USERNAME and KAGGLE_KEY to the app Secrets panel.
    The app will download the dataset automatically on first run.
"""
 
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.patches import Circle, Rectangle, Arc
from scipy.ndimage import gaussian_filter
from scipy.stats import beta
from scipy.special import betaln
 
# PAGE CONFIG
st.set_page_config(
    page_title="NBA Bayesian Shots Analysis",
    page_icon="🏀",
    layout="wide",
)
 
# KAGGLE CREDENTIALS
if "KAGGLE_USERNAME" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
if "KAGGLE_KEY" in st.secrets:
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
 
# AUTO-DOWNLOAD
_LOCAL_DEFAULT = r"C:\Users\casey\.cache\kagglehub\datasets\techbaron13\nba-shots-dataset-2001-present\versions\2\nba"
_KAGGLE_DATASET = "techbaron13/nba-shots-dataset-2001-present"
 
def _find_csv_dir(base: str) -> str:
    best_dir, best_count = base, 0
    for root, _dirs, files in os.walk(base):
        n = sum(1 for f in files if f.lower().endswith(".csv"))
        if n > best_count:
            best_count, best_dir = n, root
    return best_dir
 
@st.cache_data(show_spinner="Downloading dataset from Kaggle (first run only)...")
def _kaggle_download() -> str:
    import kagglehub
    base = kagglehub.dataset_download(_KAGGLE_DATASET)
    return _find_csv_dir(base)
 
def resolve_data_path(user_path: str) -> str:
    if os.path.isdir(user_path):
        if any(f.lower().endswith(".csv") for f in os.listdir(user_path)):
            return user_path
        found = _find_csv_dir(user_path)
        if found != user_path:
            return found
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return _kaggle_download()
    return user_path
 
# SIDEBAR
st.sidebar.title("🏀 NBA Shots Analysis")
st.sidebar.markdown("---")
DATA_PATH = st.sidebar.text_input(
    "Data folder path (local only)",
    value=_LOCAL_DEFAULT,
    help="Local path to the NBA CSV folder. On Streamlit Cloud the app downloads automatically.",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tabs:**\n"
    "1. Overview\n2. Bayesian FG%\n3. Court Heatmaps\n"
    "4. Player Rankings\n5. Era Analysis\n6. Time Analysis\n"
    "7. Projections\n8. Calibration"
)
 
# HELPERS
 
def draw_court(ax, color="black", lw=2):
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
 
 
def shot_court_heatmap(df, ax, kind="fg_pct", cmap="RdBu_r",
                       vmin=None, vmax=None, sigma=2, n_bins=60,
                       title="", min_density=0.1):
    x_bins = np.linspace(-25, 25, n_bins)
    y_bins = np.linspace(-2, 47, n_bins)
    x_b = pd.cut(df["shotX_centered"], bins=x_bins, labels=False, include_lowest=True)
    y_b = pd.cut(df["shotY_centered"], bins=y_bins, labels=False, include_lowest=True)
    mask = ~(x_b.isna() | y_b.isna())
    x_b  = x_b[mask].astype(int).values
    y_b  = y_b[mask].astype(int).values
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
        att  = gaussian_filter(attempts_grid, sigma=sigma)
        mk   = gaussian_filter(makes_grid,    sigma=sigma)
        grid = np.divide(mk, att, out=np.full_like(mk, np.nan), where=att > 0.05)
        grid[att < 0.5] = np.nan
    else:
        raise ValueError(f"Unknown kind: {kind}")
    im = ax.imshow(
        grid, origin="lower",
        extent=[x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()],
        aspect="auto", cmap=cmap, alpha=0.85, vmin=vmin, vmax=vmax,
    )
    draw_court(ax=ax, color="black", lw=2)
    ax.set_title(title)
    ax.set_xlabel("Court X")
    ax.set_ylabel("Court Y")
    return im
 
 
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
        v = (lml(1, 1, makes[:k].sum(), misses[:k].sum()) +
             lml(1, 1, makes[k:].sum(), misses[k:].sum()))
        if v > best_lml:
            best_lml, best_d = v, mids[k]
    return best_d
 
 
# KEEP COLS / DATA LOADING
_KEEP_COLS = {
    "made", "distance", "shotX", "shotY",
    "player", "team", "opp",
    "quarter", "time_remaining",
    "season", "date",
}
 
def _is_cloud() -> bool:
    return os.path.isdir("/mount/src")
 
def _read_csv_lean(filepath: str, source_name: str) -> pd.DataFrame:
    header = pd.read_csv(filepath, nrows=0)
    use    = [c for c in header.columns
              if c in _KEEP_COLS and not c.startswith("Unnamed")]
    df     = pd.read_csv(filepath, usecols=use, low_memory=False)
    df["source_file"] = source_name
    return df
 
_CLOUD_SAMPLE_FRAC = 0.02
 
@st.cache_data(show_spinner="Loading and cleaning shot data...")
def load_data(path: str):
    if not os.path.isdir(path):
        return None, []
    csv_files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
    if not csv_files:
        return None, []
    frames  = [_read_csv_lean(os.path.join(path, f), f) for f in csv_files]
    full_df = pd.concat(frames, ignore_index=True)
    del frames
 
    if _is_cloud() and _CLOUD_SAMPLE_FRAC < 1.0:
        full_df = full_df.sample(frac=_CLOUD_SAMPLE_FRAC, random_state=42).reset_index(drop=True)
 
    full_df = full_df.loc[:, ~full_df.columns.str.contains("^Unnamed")]
    for col in ("made", "distance", "shotX", "shotY"):
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors="coerce")
    for col in ("team", "opp"):
        if col in full_df.columns:
            full_df[col] = full_df[col].astype(str).str.replace("'", "", regex=False)
    if "player" in full_df.columns:
        full_df["player"] = (
            full_df["player"]
            .astype(str)
            .str.replace("'", "", regex=False)
            .str.strip()
            .str.replace(r"\s+made\s*$", "", regex=True)
        )
    full_df = full_df.dropna(subset=["made", "distance", "shotX", "shotY"]).copy()
    full_df["made"]     = full_df["made"].astype(int)
    full_df["made"]     = full_df["made"].astype("int8")
    full_df["distance"] = full_df["distance"].astype("float32")
    full_df["shotX"]    = full_df["shotX"].astype("float32")
    full_df["shotY"]    = full_df["shotY"].astype("float32")
    full_df["distance_sq"]    = full_df["distance"] ** 2
    full_df["shotX_centered"] = full_df["shotX"] - 25
    full_df["shotY_centered"] = full_df["shotY"] - 5
    s_yr = full_df["source_file"].astype(str).str.extract(r"(\d{4})", expand=False)
    full_df["season_start"] = pd.to_numeric(s_yr, errors="coerce")
    if "season" in full_df.columns:
        s2 = full_df["season"].astype(str).str.extract(r"(\d{4})", expand=False)
        yr = pd.to_numeric(s2, errors="coerce")
        full_df["season_start"] = full_df["season_start"].fillna(yr)
    if "date" in full_df.columns:
        d = pd.to_datetime(full_df["date"], errors="coerce")
        yr_from_date = np.where(d.dt.month >= 10, d.dt.year, d.dt.year - 1)
        full_df["season_start"] = full_df["season_start"].fillna(
            pd.Series(yr_from_date, index=full_df.index)
        )
    full_df["era"] = np.where(full_df["season_start"] >= 2014, "2014+", "pre-2014")
    if "time_remaining" in full_df.columns:
        def time_to_seconds(t):
            try:
                m, s = str(t).split(":")
                return int(float(m)) * 60 + int(float(s))
            except Exception:
                return np.nan
        full_df["time_sec"] = full_df["time_remaining"].apply(time_to_seconds)
        full_df["minute"]   = (12 - (full_df["time_sec"] // 60)).clip(1, 12).astype("Int64")
    else:
        full_df["time_sec"] = np.nan
        full_df["minute"]   = pd.array([pd.NA] * len(full_df), dtype="Int64")
    return full_df, csv_files
 
 
# MAIN
st.title("🏀 NBA Bayesian Shots Analysis")
st.caption("Dataset: techbaron13/nba-shots-dataset-2001-present  |  2001 - present")
 
_resolved_path = resolve_data_path(DATA_PATH)
full_df, csv_files = load_data(_resolved_path)
 
if full_df is None:
    has_creds = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if has_creds:
        st.error(
            "Could not load CSV files even after downloading from Kaggle.\n\n"
            f"Resolved path: `{_resolved_path}`\n\n"
            "Check that your KAGGLE_USERNAME and KAGGLE_KEY secrets are correct."
        )
    else:
        st.error(
            f"Could not find any CSV files at:\n\n`{DATA_PATH}`\n\n"
            "**Running locally?** Update the path in the sidebar.\n\n"
            "**Deployed on Streamlit Cloud?** Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Secrets."
        )
        st.info(
            "**How to add Kaggle secrets:**\n"
            "1. Go to your app on share.streamlit.io\n"
            "2. Click menu -> Settings -> Secrets\n"
            "3. Paste:\n"
            "```toml\n"
            "KAGGLE_USERNAME = \"your_username\"\n"
            "KAGGLE_KEY      = \"your_api_key\"\n"
            "```\n"
            "Find your key at kaggle.com -> Account -> Create New Token."
        )
    st.stop()
 
st.success(f"Loaded **{len(full_df):,}** shots from **{len(csv_files)}** files.")
 
DIST_BIN_WIDTH = 2
DIST_MAX       = full_df["distance"].max()
dist_edges     = np.arange(0, DIST_MAX + DIST_BIN_WIDTH + 1, DIST_BIN_WIDTH)
 
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Overview", "Bayesian FG%", "Court Heatmaps", "Player Rankings",
    "Era Analysis", "Time Analysis", "Projections", "Calibration",
])
 
# TAB 1 - OVERVIEW
with tab1:
    st.header("Overview")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Shots", f"{len(full_df):,}")
    col_b.metric("Made Shots",  f"{full_df['made'].sum():,}")
    col_c.metric("Overall FG%", f"{full_df['made'].mean():.1%}")
    if full_df["season_start"].notna().any():
        yr_min = int(full_df["season_start"].min())
        yr_max = int(full_df["season_start"].max())
        col_d.metric("Seasons", f"{yr_min} - {yr_max}")
    st.subheader("FG% by Shot Distance (raw)")
    fg_by_dist = full_df.groupby("distance")["made"].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(fg_by_dist.index, fg_by_dist.values)
    ax.axvline(22, color="red", ls=":", alpha=0.6, label="3-pt line (~22 ft)")
    ax.set_xlabel("Distance (ft)"); ax.set_ylabel("FG%")
    ax.set_title("FG% by Shot Distance (raw)"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with st.expander("Show data columns"):
        st.write(list(full_df.columns))
    with st.expander("Show sample rows"):
        st.dataframe(full_df.head(20))
 
# TAB 2 - BAYESIAN FG%
with tab2:
    st.header("Bayesian FG% by Distance")
    ALPHA = 1; BETA_PRIOR = 1
    summary_df = fit_bayes_distance(full_df, dist_edges)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary_df["dist_mid"], summary_df["post_mean"], lw=2, label="Posterior Mean FG%")
    ax.fill_between(summary_df["dist_mid"], summary_df["ci_lower"], summary_df["ci_upper"],
                    alpha=0.25, label="95% Credible Interval")
    ax.axvline(22, color="red", ls=":", alpha=0.6, label="3-pt line")
    ax.set_xlabel("Distance (ft)"); ax.set_ylabel("Shot Make Probability")
    ax.set_title("Bayesian FG% by Distance"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
    st.subheader("Bootstrap Reliability Check")
    n_boot_slider = st.slider("Bootstrap iterations", 100, 1000, 500, step=100)
 
    @st.cache_data(show_spinner="Running bootstrap...")
    def run_bootstrap(made_bytes, codes_bytes, n_bins_total, n_boot, mids_bytes):
        made_arr  = np.frombuffer(made_bytes,  dtype=np.float64)
        bin_codes = np.frombuffer(codes_bytes, dtype=np.int64)
        dist_mids = np.frombuffer(mids_bytes,  dtype=np.float64)
        N   = len(made_arr); RNG = np.random.default_rng(42)
        bpm = np.empty((n_boot, n_bins_total))
        for b in range(n_boot):
            idx = RNG.integers(0, N, size=N)
            mk  = np.bincount(bin_codes[idx], weights=made_arr[idx], minlength=n_bins_total)
            at  = np.bincount(bin_codes[idx], minlength=n_bins_total)
            bpm[b] = (ALPHA + mk) / (ALPHA + BETA_PRIOR + at)
        return pd.DataFrame({
            "dist_mid":   dist_mids,
            "boot_lower": np.quantile(bpm, 0.025, axis=0),
            "boot_upper": np.quantile(bpm, 0.975, axis=0),
            "boot_mean":  bpm.mean(axis=0),
        })
 
    bayes_df  = full_df[["made", "distance"]].dropna().copy()
    bayes_df["dist_bin"] = pd.cut(bayes_df["distance"], bins=dist_edges, right=False)
    bin_codes    = pd.Categorical(bayes_df["dist_bin"]).codes
    n_bins_total = int(bin_codes.max()) + 1
    made_arr     = bayes_df["made"].to_numpy(dtype=np.float64)
 
    boot_band = run_bootstrap(
        made_arr.tobytes(), bin_codes.astype(np.int64).tobytes(),
        n_bins_total, n_boot_slider,
        summary_df["dist_mid"].to_numpy(dtype=np.float64).tobytes(),
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary_df["dist_mid"], summary_df["post_mean"], lw=2, label="Posterior Mean")
    ax.fill_between(summary_df["dist_mid"], summary_df["ci_lower"], summary_df["ci_upper"],
                    alpha=0.20, label="Bayesian 95% CrI")
    ax.fill_between(boot_band["dist_mid"], boot_band["boot_lower"], boot_band["boot_upper"],
                    alpha=0.20, label=f"Bootstrap 95% CI (n={n_boot_slider})")
    ax.axvline(22, color="red", ls=":", alpha=0.6, label="3-pt line")
    ax.set_xlabel("Distance (ft)"); ax.set_ylabel("Shot Make Probability")
    ax.set_title("Bayesian FG% with Bootstrap Reliability Check"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
    with st.expander("Summary table"):
        st.dataframe(
            summary_df[["dist_mid","attempts","makes","post_mean","ci_lower","ci_upper"]]
            .rename(columns={"dist_mid":"Distance (ft)","post_mean":"Post. Mean",
                             "ci_lower":"CI Low","ci_upper":"CI High"}).round(3)
        )
 
# TAB 3 - COURT HEATMAPS
with tab3:
    st.header("Court Heatmaps")
    st.subheader("Splash Bros -- KDE of Made Shots")
    sg_players = ["Stephen Curry", "Klay Thompson"]
    sg_df   = full_df[full_df["player"].isin(sg_players)].copy()
    sg_made = sg_df[sg_df["made"] == 1]
    if sg_made.empty:
        st.warning("No Splash Bros shots found.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.kdeplot(data=sg_made, x="shotX_centered", y="shotY_centered",
                    fill=True, cmap="Reds", levels=50, thresh=0.05, ax=ax)
        ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
        ax.set_title("Splash Bros -- Made Shots (KDE, centered)")
        ax.set_xlabel("Court X (centered)"); ax.set_ylabel("Court Y (centered)")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
    st.subheader("Splash Bros -- Shot Efficiency Heatmap")
    if not sg_df.empty:
        N_BINS = 60
        x_bins = np.linspace(-25, 25, N_BINS); y_bins = np.linspace(-2, 47, N_BINS)
        plot_df = sg_df.copy()
        plot_df["x_bin"] = pd.cut(plot_df["shotX_centered"], bins=x_bins, labels=False, include_lowest=True)
        plot_df["y_bin"] = pd.cut(plot_df["shotY_centered"], bins=y_bins, labels=False, include_lowest=True)
        plot_df = plot_df.dropna(subset=["x_bin","y_bin"]).astype({"x_bin":int,"y_bin":int})
        n_y, n_x = N_BINS-1, N_BINS-1
        makes_grid = np.zeros((n_y,n_x)); attempts_grid = np.zeros((n_y,n_x))
        np.add.at(attempts_grid, (plot_df["y_bin"].values, plot_df["x_bin"].values), 1)
        np.add.at(makes_grid,    (plot_df["y_bin"].values, plot_df["x_bin"].values), plot_df["made"].values)
        mk_s  = gaussian_filter(makes_grid, sigma=2)
        att_s = gaussian_filter(attempts_grid, sigma=2)
        fg_sm = np.divide(mk_s, att_s, out=np.full_like(mk_s, np.nan), where=att_s > 0.05)
        fg_sm[att_s < 0.5] = np.nan
        fig, ax = plt.subplots(figsize=(8,7))
        im = ax.imshow(fg_sm, origin="lower",
                       extent=[x_bins.min(),x_bins.max(),y_bins.min(),y_bins.max()],
                       aspect="auto", cmap="RdBu_r", alpha=0.9, vmin=0.3, vmax=0.7)
        draw_court(ax=ax); plt.colorbar(im, ax=ax, label="Field Goal %")
        ax.set_title("Splash Bros -- Shot Efficiency (Gaussian smoothed)")
        ax.set_xlabel("Court X"); ax.set_ylabel("Court Y")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
    st.subheader("FG% Heatmap -- Pre-2014 vs 2014+")
    fig, axes = plt.subplots(1, 2, figsize=(16,7))
    for ax, era in zip(axes, ["pre-2014","2014+"]):
        sub = full_df[full_df["era"] == era]
        im  = shot_court_heatmap(sub, ax=ax, kind="fg_pct", cmap="RdBu_r", sigma=4,
                                 vmin=0.3, vmax=0.7, title=f"FG% -- {era}  (n={len(sub):,})")
        plt.colorbar(im, ax=ax, label="FG%")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
# TAB 4 - PLAYER RANKINGS
with tab4:
    st.header("Player Rankings")
    st.subheader("Top Three-Point Shooters (22-26 ft)")
    min_attempts = st.slider("Minimum attempts", 100, 2000, 1000, step=100, key="3pt_min")
 
    if all(c in full_df.columns for c in ["player","distance","made"]):
        p3 = (
            full_df[full_df["distance"].between(22,26)]
            .groupby("player")
            .agg(attempts=("made","size"), makes=("made","sum"))
            .query(f"attempts >= {min_attempts}")
            .copy()
        )
        p3["post_mean"] = (1 + p3["makes"]) / (2 + p3["attempts"])
        p3["ci_lower"]  = beta.ppf(0.025, 1+p3["makes"], 1+p3["attempts"]-p3["makes"])
        p3["ci_upper"]  = beta.ppf(0.975, 1+p3["makes"], 1+p3["attempts"]-p3["makes"])
        p3 = p3.sort_values("post_mean", ascending=False)
        top_n = st.slider("Show top N players", 10, 30, 20)
        top_k = p3.head(top_n)
        fig, ax = plt.subplots(figsize=(10, max(5, top_n//2+2)))
        ax.barh(top_k.index[::-1], top_k["post_mean"][::-1],
                xerr=[top_k["post_mean"][::-1]-top_k["ci_lower"][::-1],
                      top_k["ci_upper"][::-1]-top_k["post_mean"][::-1]],
                color="C3", alpha=0.8, capsize=4)
        ax.set_xlabel("Posterior FG% (22-26 ft)")
        ax.set_title(f"Top {top_n} Three-Point Shooters (min {min_attempts} attempts)")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with st.expander("Full table"):
            st.dataframe(p3.head(50).round(3))
    else:
        st.warning("Player or distance column not found.")
 
    st.subheader("Elite Scorers (>20 PPG estimate)")
    PPG_THRESHOLD   = st.slider("PPG threshold", 10, 30, 20, key="ppg_thresh")
    MIN_TOTAL_SHOTS = st.slider("Min total shots", 200, 5000, 1000, step=100, key="min_shots")
    full_df["is_three"]     = (((full_df["shotX_centered"].abs() >= 22) & (full_df["shotY_centered"] <= 14))
                                | (full_df["distance"] >= 23.75))
    full_df["points_value"] = np.where(full_df["is_three"], 3, 2)
    full_df["points"]       = full_df["made"] * full_df["points_value"]
    GAME_CANDIDATES = ["match_id","game_id","game","game_date","date","matchup_id","GAME_ID"]
    game_col = next((c for c in GAME_CANDIDATES if c in full_df.columns), None)
    if game_col:
        ppg = full_df.groupby("player").agg(
            total_pts=("points","sum"), games=(game_col,"nunique"), total_shots=("made","size"))
    else:
        ppg = full_df.groupby("player").agg(total_pts=("points","sum"), total_shots=("made","size"))
        ppg["games"] = ppg["total_shots"] / 10
    ppg["ppg"] = ppg["total_pts"] / ppg["games"]
    elite = ppg[(ppg["ppg"] > PPG_THRESHOLD) & (ppg["total_shots"] >= MIN_TOTAL_SHOTS)].sort_values("ppg", ascending=False)
    elite_players = elite.index.tolist()
    st.metric("Players above threshold", len(elite))
    st.dataframe(elite.head(20).round(2))
 
# TAB 5 - ERA ANALYSIS
with tab5:
    st.header("Era Analysis: Pre-2014 vs 2014+")
    ERA_SPLIT = st.number_input("Era split year", min_value=2001, max_value=2023, value=2014, step=1)
    full_df["era"] = np.where(full_df["season_start"] >= ERA_SPLIT, f"{ERA_SPLIT}+", f"pre-{ERA_SPLIT}")
    eras = sorted(full_df["era"].dropna().unique())
    era_summaries    = {era: add_ep(fit_bayes_distance(full_df[full_df["era"]==era], dist_edges)) for era in eras}
    era_change_points= {era: map_change_point(s) for era, s in era_summaries.items()}
    colors = {eras[0]:"C0", eras[1]:"C3"} if len(eras)>=2 else {}
 
    fig, ax = plt.subplots(figsize=(10,5))
    for era, s in era_summaries.items():
        c = colors.get(era)
        ax.plot(s["dist_mid"], s["post_mean"], lw=2, color=c, label=f"{era} FG%")
        ax.fill_between(s["dist_mid"], s["ci_lower"], s["ci_upper"], alpha=0.2, color=c)
    ax.axvline(22.0, color="gray", ls=":", label="3-pt line")
    ax.set_xlabel("Distance (ft)"); ax.set_ylabel("FG%")
    ax.set_title(f"Posterior FG% by Distance -- Pre vs Post {ERA_SPLIT}"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
    st.subheader("Expected Points per Shot by Distance")
    fig, ax = plt.subplots(figsize=(10,5))
    for era, s in era_summaries.items():
        c = colors.get(era)
        ax.plot(s["dist_mid"], s["ep_mean"], lw=2, color=c, label=f"{era} EP")
        ax.fill_between(s["dist_mid"], s["ep_lower"], s["ep_upper"], alpha=0.2, color=c)
        cp = era_change_points[era]
        if cp is not None:
            ax.axvline(cp, ls="--", color=c, alpha=0.7, label=f"{era} CP: {cp:.1f} ft")
    ax.axvline(22.0, color="gray", ls=":", label="3-pt line")
    ax.set_xlabel("Distance (ft)"); ax.set_ylabel("EP per Shot")
    ax.set_title(f"Expected Points -- Pre vs Post {ERA_SPLIT}"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
    st.subheader("Shot-Attempt Distribution by Distance")
    fig, ax = plt.subplots(figsize=(10,5))
    for era, s in era_summaries.items():
        ax.plot(s["dist_mid"], s["attempts"]/s["attempts"].sum(), lw=2,
                color=colors.get(era), label=f"{era} attempt share")
    ax.axvline(22.0, color="gray", ls=":", label="3-pt line")
    ax.set_xlabel("Distance (ft)"); ax.set_ylabel("Share of all attempts")
    ax.set_title(f"Shot Distribution -- Pre vs Post {ERA_SPLIT}"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
    st.subheader("Era Summary")
    rows = []
    for era, s in era_summaries.items():
        cp = era_change_points[era]
        rows.append({"Era":era,"Shots":f"{s['attempts'].sum():,}",
                     "FG%":f"{s['makes'].sum()/s['attempts'].sum():.3f}",
                     "Change-point (ft)":f"{cp:.1f}" if cp else "--"})
    st.table(pd.DataFrame(rows))
 
    st.subheader("Bayesian Change-Point Detection")
    for era, s in era_summaries.items():
        mc = s["makes"].values; ms = s["misses"].values; md = s["dist_mid"].values
        null_lml = betaln(1+mc.sum(), 1+ms.sum()) - betaln(1,1)
        records = []
        for k in range(2, len(md)-1):
            l = betaln(1+mc[:k].sum(),1+ms[:k].sum()) - betaln(1,1)
            r = betaln(1+mc[k:].sum(),1+ms[k:].sum()) - betaln(1,1)
            records.append({"dist":md[k],"log_marginal":l+r})
        if not records:
            continue
        cp_df    = pd.DataFrame(records)
        best_idx = cp_df["log_marginal"].idxmax()
        best_d   = cp_df.loc[best_idx,"dist"]
        best_lml = cp_df.loc[best_idx,"log_marginal"]
        bf       = np.exp(np.clip(best_lml - null_lml, -700, 700))
        st.markdown(f"**{era}** -- MAP change-point: **{best_d:.1f} ft** | Bayes factor: **{bf:.2e}**")
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(cp_df["dist"], cp_df["log_marginal"], color=colors.get(era,"C0"))
        ax.axvline(best_d, color="red", ls="--", label=f"MAP: {best_d:.1f} ft")
        ax.set_xlabel("Candidate change-point (ft)"); ax.set_ylabel("Joint log marginal likelihood")
        ax.set_title(f"Change-point scan -- {era}"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
# TAB 6 - TIME ANALYSIS
with tab6:
    st.header("Time-of-Quarter Analysis")
    if full_df["time_sec"].isna().all():
        st.warning("No time_remaining column found -- time analysis unavailable.")
    elif "quarter" not in full_df.columns:
        st.warning("No quarter column found -- time analysis unavailable.")
    else:
        q_avail = full_df["quarter"].dropna().unique().tolist()
        sel_q   = st.multiselect("Select quarters", q_avail,
                                 default=[q for q in ["1st quarter","2nd quarter","3rd quarter","4th quarter"]
                                          if q in q_avail])
        full_df["dist_bin_time"] = pd.cut(full_df["distance"], bins=dist_edges, right=False)
        for q in sel_q:
            q_df = full_df[full_df["quarter"]==q].copy()
            if q_df.empty:
                continue
            g = (q_df.groupby(["dist_bin_time","minute"], observed=False)
                 .agg(attempts=("made","size"),makes=("made","sum")).reset_index())
            g["post_mean"] = (1+g["makes"])/(2+g["attempts"])
            pivot = g.pivot(index="minute", columns="dist_bin_time", values="post_mean")
            pivot.columns = [f"{int(b.left)}-{int(b.right)}" for b in pivot.columns]
            fig, ax = plt.subplots(figsize=(14,5))
            sns.heatmap(pivot, cmap="RdBu_r", vmin=0.3, vmax=0.7, ax=ax)
            ax.set_title(f"Posterior FG% by Distance and Minute ({q})")
            ax.set_xlabel("Distance (ft)"); ax.set_ylabel("Minute of Quarter")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
    st.subheader("Quarter Heatmaps -- Elite Scorers (post-2014)")
    if "quarter" not in full_df.columns:
        st.info("Quarter column not available.")
    elif not elite_players:
        st.info("No elite players found with current thresholds (check Player Rankings tab).")
    else:
        ep_post = full_df[(full_df["era"]==f"{ERA_SPLIT}+") & (full_df["player"].isin(elite_players))]
        q_list  = [q for q in ["1st quarter","2nd quarter","3rd quarter","4th quarter"]
                   if q in full_df["quarter"].unique()]
        if q_list:
            fig, axes = plt.subplots(2, 2, figsize=(16,14))
            for ax, q in zip(axes.flat, q_list):
                sub = ep_post[ep_post["quarter"]==q]
                im  = shot_court_heatmap(sub, ax=ax, kind="fg_pct", cmap="RdBu_r", sigma=2,
                                         vmin=0.0, vmax=0.5, title=f"{q} -- elite scorers\nn={len(sub):,}")
                plt.colorbar(im, ax=ax, label="FG%")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
        st.subheader("Clutch FG% -- Last 5 Min of Q4")
        if "4th quarter" in full_df["quarter"].unique():
            clutch = full_df[(full_df["era"]==f"{ERA_SPLIT}+") &
                             (full_df["player"].isin(elite_players)) &
                             (full_df["quarter"]=="4th quarter") &
                             (full_df["time_sec"]<=300)]
            if not clutch.empty:
                fig, ax = plt.subplots(figsize=(9,8))
                im = shot_court_heatmap(clutch, ax=ax, kind="fg_pct", cmap="RdBu_r", sigma=2,
                                        vmin=0.0, vmax=0.5, title=f"Clutch FG% n={len(clutch):,}")
                plt.colorbar(im, ax=ax, label="FG%")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
# TAB 7 - PROJECTIONS
with tab7:
    st.header("Projections & Expected Points")
    st.subheader("Expected Points Curve")
    THREE_PT_LINE = st.slider("3-pt line distance (ft)", 20.0, 25.0, 22.0, step=0.25)
    ep = add_ep(summary_df, three_pt_line=THREE_PT_LINE)
    league_ep = (ep["ep_mean"] * ep["attempts"]).sum() / ep["attempts"].sum()
    above     = ep[ep["ep_upper"] >= league_ep]
    threshold = above["dist_mid"].max() if not above.empty else float("nan")
    col1, col2 = st.columns(2)
    col1.metric("League-avg EP (pts/shot)", f"{league_ep:.3f}")
    col2.metric("Stop-shooting threshold", f"{threshold:.1f} ft" if np.isfinite(threshold) else "N/A")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(ep["dist_mid"], ep["ep_mean"], lw=2, color="C0", label="Posterior EP")
    ax.fill_between(ep["dist_mid"], ep["ep_lower"], ep["ep_upper"], alpha=0.25, color="C0",
                    label="95% Credible Interval")
    ax.axhline(league_ep, color="gray", ls="--", label=f"League avg EP ~{league_ep:.2f}")
    ax.axvline(THREE_PT_LINE, color="red", ls=":", alpha=0.7, label="3-pt line")
    if np.isfinite(threshold):
        ax.axvline(threshold, color="black", ls="-.", alpha=0.6, label=f"Threshold ~{threshold:.1f} ft")
    ax.set_xlabel("Distance (ft)"); ax.set_ylabel("Expected Points per Shot")
    ax.set_title("Expected Points by Shot Distance"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
 
    st.subheader("FG% Projection by Distance")
    PROJECTION_YEAR = st.slider("Projection year", 2025, 2040, 2030)
 
    @st.cache_data(show_spinner="Building projection model...")
    def compute_projection(df_json, edges_key, proj_year):
        df    = pd.read_json(io.StringIO(df_json), orient="split")
        edges = np.arange(0, edges_key + DIST_BIN_WIDTH + 1, DIST_BIN_WIDTH)
        per_s = (df.dropna(subset=["season_start"])
                 .assign(dist_bin=lambda d: pd.cut(d["distance"], bins=edges, right=False))
                 .groupby(["season_start","dist_bin"], observed=False)
                 .agg(attempts=("made","size"),makes=("made","sum")).reset_index())
        per_s["fg_pct"] = per_s["makes"] / per_s["attempts"]
        per_s = per_s[per_s["attempts"] >= 100]
        last  = df["season_start"].max(); span = df["season_start"].max() - df["season_start"].min()
        proj  = []
        for bl, g in per_s.groupby("dist_bin", observed=True):
            if len(g) < 5: continue
            x = g["season_start"].values.astype(float); y = g["fg_pct"].values
            if np.std(x) < 1e-6: continue
            slope, intercept = np.polyfit(x, y, 1)
            pred = float(np.clip(slope*proj_year+intercept, 0.02, 0.95))
            res  = y - (slope*x+intercept)
            sig  = np.std(res, ddof=2) if len(g)>2 else 0.05
            band = sig * (1 + max(proj_year-last,0)/max(span,1)) * 1.96
            proj.append({"dist_bin":bl,
                         "dist_mid":float(bl.left+(bl.right-bl.left)/2),
                         "pred":pred,
                         "lo":float(np.clip(pred-band,0,1)),
                         "hi":float(np.clip(pred+band,0,1))})
        return pd.DataFrame(proj).sort_values("dist_mid")
 
    needed = ["made","distance","season_start"]
    if all(c in full_df.columns for c in needed) and full_df["season_start"].notna().any():
        proj_df = compute_projection(
            full_df[needed].to_json(orient="split"), float(DIST_MAX), PROJECTION_YEAR)
        pre_s  = fit_bayes_distance(full_df[full_df["era"]=="pre-2014"], dist_edges)
        post_s = fit_bayes_distance(full_df[full_df["era"]=="2014+"],    dist_edges)
        fig, ax = plt.subplots(figsize=(11,5))
        ax.plot(pre_s["dist_mid"],  pre_s["post_mean"],  lw=2, color="C0", label="pre-2014")
        ax.fill_between(pre_s["dist_mid"], pre_s["ci_lower"], pre_s["ci_upper"], alpha=0.2, color="C0")
        ax.plot(post_s["dist_mid"], post_s["post_mean"], lw=2, color="C3", label="2014+")
        ax.fill_between(post_s["dist_mid"], post_s["ci_lower"], post_s["ci_upper"], alpha=0.2, color="C3")
        ax.plot(proj_df["dist_mid"], proj_df["pred"], lw=2, color="C2", ls="--",
                label=f"{PROJECTION_YEAR} projection")
        ax.fill_between(proj_df["dist_mid"], proj_df["lo"], proj_df["hi"],
                        alpha=0.2, color="C2", label=f"{PROJECTION_YEAR} 95% band")
        ax.set_xlabel("Distance (ft)"); ax.set_ylabel("FG%")
        ax.set_title(f"FG% by Distance -- Pre-2014, 2014+, and {PROJECTION_YEAR} Projection"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Projections are naive linear extrapolations -- not real forecasts.")
    else:
        st.warning("Season/year data not available for projection modeling.")
 
# TAB 8 - CALIBRATION
with tab8:
    st.header("Calibration & Out-of-Sample Validation")
    n_test_seasons = st.slider("Hold-out seasons (most recent)", 1, 5, 2)
    valid       = full_df.dropna(subset=["season_start"]).copy()
    all_seasons = sorted(valid["season_start"].dropna().unique())
    if len(all_seasons) < n_test_seasons + 3:
        st.warning("Not enough distinct seasons for this split.")
    else:
        test_seasons  = all_seasons[-n_test_seasons:]
        train_seasons = [s for s in all_seasons if s not in test_seasons]
        train = valid[valid["season_start"].isin(train_seasons)].copy()
        test  = valid[valid["season_start"].isin(test_seasons)].copy()
        st.markdown(f"**Train:** {len(train):,} shots ({len(train_seasons)} seasons)  |  "
                    f"**Test:** {len(test):,} shots (seasons {test_seasons})")
        train_summary    = fit_bayes_distance(train, dist_edges)
        prob_lookup      = dict(zip(pd.Categorical(train_summary["dist_bin"]).codes,
                                    train_summary["post_mean"]))
        overall_train_fg = float(train["made"].mean())
        test["dist_bin"] = pd.cut(test["distance"], bins=dist_edges, right=False)
        test_bin_codes   = pd.Categorical(test["dist_bin"], categories=train_summary["dist_bin"]).codes
        test["pred"]     = pd.Series(test_bin_codes, index=test.index).map(prob_lookup)
        test["pred"]     = test["pred"].fillna(overall_train_fg)
        test = test.dropna(subset=["pred","made"])
        y_true = test["made"].values.astype(float)
        y_pred = test["pred"].values.astype(float)
        y_base = np.full_like(y_true, overall_train_fg)
        def brier(y, p):  return np.mean((p-y)**2)
        def logloss(y, p):
            p = np.clip(p, 1e-6, 1-1e-6)
            return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
        skill = 1 - brier(y_true, y_pred) / brier(y_true, y_base)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Brier (model)",    f"{brier(y_true,y_pred):.4f}")
        m2.metric("Brier (baseline)", f"{brier(y_true,y_base):.4f}")
        m3.metric("Log-loss (model)", f"{logloss(y_true,y_pred):.4f}")
        m4.metric("Brier skill score", f"{skill:.4f}", delta="vs baseline",
                  delta_color="normal" if skill > 0 else "inverse")
        N_DECILES = 10
        deciles = pd.qcut(y_pred, q=N_DECILES, duplicates="drop")
        cal = (pd.DataFrame({"y":y_true,"p":y_pred,"bin":deciles})
               .groupby("bin", observed=True)
               .agg(predicted=("p","mean"),observed=("y","mean"),n=("y","size"))
               .reset_index(drop=True))
        fig, ax = plt.subplots(figsize=(7,7))
        ax.plot([0,1],[0,1],"k--",alpha=0.5,label="Perfect calibration")
        ax.scatter(cal["predicted"],cal["observed"],s=80,color="C3",zorder=5,label="Decile bins")
        for _, r in cal.iterrows():
            ax.annotate(f"n={int(r['n']):,}",(r["predicted"],r["observed"]),
                        fontsize=8,xytext=(5,5),textcoords="offset points")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel("Predicted FG%"); ax.set_ylabel("Observed FG%")
        ax.set_title("Calibration Plot -- held-out test seasons")
        ax.legend(); ax.set_aspect("equal")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        test_by_bin = (test.groupby("dist_bin", observed=False)
                       .agg(observed=("made","mean"),predicted=("pred","mean"),n=("made","size"))
                       .reset_index())
        test_by_bin["dist_mid"] = test_by_bin["dist_bin"].apply(
            lambda b: b.left+(b.right-b.left)/2 if isinstance(b,pd.Interval) else np.nan)
        test_by_bin = test_by_bin.dropna(subset=["dist_mid","observed"])
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(test_by_bin["dist_mid"],test_by_bin["predicted"],"o-",
                label="Predicted (train model)",color="C0")
        ax.plot(test_by_bin["dist_mid"],test_by_bin["observed"],"s-",
                label="Observed (test seasons)",color="C3")
        ax.set_xlabel("Distance (ft)"); ax.set_ylabel("FG%")
        ax.set_title("Predicted vs Observed FG% by Distance -- Held-out Seasons"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with st.expander("Calibration table"):
            st.dataframe(cal.round(3))
