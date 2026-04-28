NBA Bayesian Shots Analysis

Casey Campbell | Statistical & Machine Learning | 2026

PROJECT DESCRIPTION
This project applies Bayesian statistical modeling to a comprehensive
shot-by-shot NBA dataset spanning 2000-2022. The goal is to move beyond
raw field goal percentages and use principled probabilistic methods to
understand shooting efficiency across distance, court zone, player, era,
and game situation.
Key analyses include:

Bayesian FG% by distance: Beta-Binomial conjugate model with 95%
credible intervals, showing how shot-make probability declines with
distance and where the inflection points are.

Bootstrap reliability check: Vectorized bootstrap resampling to
cross-validate the Bayesian credible intervals.
Bayesian change-point detection: Grid search over joint log marginal
likelihood to identify the MAP change-point in FG% by distance
(found consistently at ~5 ft, the boundary between layup and
jump-shot efficiency).

Expected Points (EP) analysis: Combines posterior FG% with point
value (2 or 3) to find the stop-shooting threshold (~27 ft) beyond
which shots yield less than the league average of ~1.01 pts/shot.
Era analysis (pre-2014 vs. 2014+): Compares FG%, expected points,
shot distribution, and change-points across the analytics revolution
split, showing how post-2014 players extract more value from
3-point range.

Court heatmaps: Gaussian-smoothed FG% heatmaps by court zone,
including Splash Bros (Stephen Curry & Klay Thompson) efficiency
profiles and pre/post-2014 era comparisons.
Player rankings: Bayesian posterior FG% estimates for three-point
shooters (22-26 ft, min. 1,000 attempts), with credible intervals
to reflect uncertainty rather than raw sample percentages.

Clutch analysis: FG% by court zone in the final 5 minutes of Q4
for elite scorers (>20 PPG, post-2014), finding that elite players
do not shift zones under pressure.

Time-of-quarter heatmaps: Posterior FG% by distance and minute of
quarter across all four quarters.

2030 FG% projections: Linear extrapolation of per-season posterior
estimates with 95% extrapolation bands (directional trend indicator,
not a forecast).

Calibration & out-of-sample validation: Temporal train/test split
on held-out seasons, Brier score, log-loss, and a calibration plot
to confirm the model is well-calibrated across the FG% range.

The project is delivered in two forms: a Python script (dsci441proj.py)
that contains all compiled analysis code and can be run top-to-bottom
in Spyder, and an interactive Streamlit web application (app.py) with
eight tabs covering each analysis area.

DATA SOURCE
Dataset:  NBA Shots Dataset 2001-Present
Source:   Kaggle -- techbaron13/nba-shots-dataset-2001-present
URL:      https://www.kaggle.com/datasets/techbaron13/nba-shots-dataset-2001-present
Coverage: Shot-by-shot records from the 2000-01 season through 2022
Format:   Multiple CSV files, one per season, stored in an nba/ subfolder
Key columns used:
made            1 = made, 0 = missed
distance        Shot distance in feet
shotX / shotY   Court coordinates
player          Player name
quarter         Quarter of the game
time_remaining  Time left in the quarter (MM:SS)
season / date   Used to assign season year


Downloading the Data:
Locally -- Download via kagglehub or the Kaggle CLI:
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('techbaron13/nba-shots-dataset-2001-present')"
The data will be cached at:
~/.cache/kagglehub/datasets/techbaron13/nba-shots-dataset-2001-present/versions/2/nba/
On Streamlit Cloud -- The app downloads the dataset automatically on
first run using your Kaggle API credentials (see deployment instructions
below).


REQUIRED PACKAGES
Install all dependencies with:
pip install -r requirements.txt
Contents of requirements.txt:
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
kagglehub>=0.2.0

Package descriptions:
streamlit    Web application framework
pandas       Data loading and manipulation
numpy        Numerical computing, grid operations
matplotlib   All charts and court diagrams
seaborn      KDE heatmaps, quarter heatmaps
scipy        Beta distribution (credible intervals), Gaussian smoothing,
special functions
kagglehub    Automatic dataset download on Streamlit Cloud
HOW TO RUN
Option 1: Running the Script in Spyder
The main analysis script (dsci441proj.py) contains all compiled code and
is designed to run top-to-bottom in Spyder. Each section is clearly
delimited and can be run independently using Spyder's "Run cell" feature
(Ctrl+Enter).

Install dependencies:
pip install pandas numpy matplotlib seaborn scipy kagglehub
Download the dataset:
python -c "import kagglehub; kagglehub.dataset_download('techbaron13/nba-shots-dataset-2001-present')"
Update the data path at the top of the script to match your local
cache:
PATH = r"C:\Users<your-username>.cache\kagglehub\datasets\techbaron13\nba-shots-dataset-2001-present\versions\2\nba"
Open dsci441proj.py in Spyder and run top-to-bottom. Each section
produces one or more inline plots. The script is organized into the
following pipeline:
Section 1   Load & Clean          Reads all CSVs, coerces types,
engineers features
Section 2   FG% by Distance       Raw field goal percentage vs.
distance curve
Section 3   Splash Bros KDE       KDE heatmap of Curry & Thompson
made shots
Section 4   Court Drawing Helper  draw_court() utility function
Section 5   Shot Efficiency       Gaussian-smoothed FG% heatmap
Heatmap               (Splash Bros)
Section 6   Bayesian              Posterior FG% with 95% credible
Beta-Binomial         intervals by distance
Section 7   Bootstrap Check       Vectorized bootstrap to
cross-validate credible intervals
Section 7B  Player Rankings       Top 3PT shooters by Bayesian
posterior FG%
Section 8   Time-of-Quarter       FG% by distance x minute for
Heatmaps              each quarter
EP Curve                          Expected points per shot +
stop-shooting threshold
Change-Point Detection            MAP change-point in FG%(distance)
via Bayes factor
Era Analysis                      Pre-2014 vs. 2014+ FG%, EP,
attempt distribution
Calibration                       Temporal train/test split, Brier
score, calibration plot
Court Heatmaps                    Era FG% heatmaps, elite scorer
quarterly and clutch maps
2030 Projection                   Linear extrapolation of
per-season FG% trends

Option 2: Running the Streamlit App Locally

Clone the repository:
git clone https://github.com/your-username/dsci441-semester-project.git
cd dsci441-semester-project
Install dependencies:
pip install -r requirements.txt
Download the dataset (if you haven't already):
python -c "import kagglehub; kagglehub.dataset_download('techbaron13/nba-shots-dataset-2001-present')"
Launch the app:
streamlit run app.py
Set the data path in the sidebar to the folder containing the CSV
files. The default path is:
C:\Users\casey.cache\kagglehub\datasets\techbaron13\nba-shots-dataset-2001-present\versions\2\nba
Update this to match your local cache location if different.

Option 3: Deploying on Streamlit Community Cloud
The app is configured to download the dataset automatically on Streamlit
Cloud using your Kaggle API credentials.

Push app.py and requirements.txt to a GitHub repository.
Go to share.streamlit.io and click "New app". Connect your GitHub
repo and set the main file to app.py.
Before deploying, add your Kaggle credentials as secrets:

Click Settings > Secrets on your app dashboard
Paste the following (with your real values):
KAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY      = "your_kaggle_api_key"

Find your API key at kaggle.com > Account > Create New Token.
Click Deploy. On first boot the app will download and extract the
dataset (~93 MB), which takes about 30-60 seconds. Subsequent boots
use the cached data.

NOTE: Streamlit Community Cloud free tier has a 1 GB RAM limit. The app
loads only the columns it needs and samples 50% of rows on cloud to stay
within this limit. All statistical results remain valid given the dataset
size.


Note: Streamlit Community Cloud free tier has a 1 GB RAM limit. The app loads only the columns it needs and samples 50% of rows on cloud to stay within this limit. All statistical results remain valid given the dataset size.
