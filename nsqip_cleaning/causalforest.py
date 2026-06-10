"""
Causal inference analysis for Specific Aim 2 (SA2) procedural volume,
using a LinearDML model to estimate the causal effect of wRVU revaluation
on each CPT code's relative share of all NSQIP cases per year.

Relative share (solo_share) is used as the outcome rather than raw counts
to control for NSQIP registry growth over time — the denominator is all
NSQIP cases across all specialties in a given year, not just ENT cases,
so that secular growth in NSQIP participation does not confound the
revaluation effect estimate.

Requires:
    - data/nsqip/combined_filtered.csv   main NSQIP analysis file
    - data/final_CPT_1.csv               ARD features including Most Recent
                                         RUC Review date (used as revaluation year)
    - data/nsqip/                        directory of yearly raw NSQIP CSVs
                                         (used to compute total_nsqip_cases
                                         per year as the volume denominator)

Pipeline:
    1. get_total_nsqip_per_year()   streams yearly CSVs to count all NSQIP
                                    cases per year (denominator)
    2. build_volume_panel()         builds CPT-year panel with solo_share
                                    outcome, ARD features, treatment indicator
                                    (post-revaluation within ±5 year window)
    3. run_volume_causal_forest()   fits LinearDML with gradient boosting
                                    nuisance models; treatment = post-revaluation,
                                    outcome = solo_share, features = ARD
                                    procedure characteristics

Output:
    Prints panel summary and fitted model to console.
    Extend main() to call est.effect() and save treatment effect CSVs
    as needed.

Dependencies:
    pip install pandas numpy scikit-learn econml
"""

import pandas as pd
import numpy as np
import os, glob

from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from data_handling_nsqip import standardize_cpt


# =========================================================
# TOTAL NSQIP CASES PER YEAR (STREAMING)
# =========================================================
def get_total_nsqip_per_year(nsqip_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(nsqip_dir, "*.csv"))

    frames = []
    for f in files:
        df = pd.read_csv(f, usecols=["PUFYEAR"], low_memory=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    return (
        combined.groupby("PUFYEAR")
        .size()
        .reset_index(name="total_nsqip_cases")
    )


# =========================================================
# VOLUME PANEL ONLY
# =========================================================
def build_volume_panel(filtered_df, ard_df, nsqip_dir):

    filtered_df = filtered_df.copy()
    ard_df = ard_df.copy()

    filtered_df["CPT"] = standardize_cpt(filtered_df["CPT"])
    ard_df["CPT"] = standardize_cpt(ard_df["CPT"])

    # numerator: CPT-year counts (filtered cohort)
    filtered_counts = (
        filtered_df.groupby(["CPT", "PUFYEAR"])
        .size()
        .reset_index(name="filtered_count")
    )

    # denominator: all NSQIP cases per year
    total_per_year = get_total_nsqip_per_year(nsqip_dir)

    panel = filtered_counts.merge(total_per_year, on="PUFYEAR", how="left")

    panel["solo_share"] = panel["filtered_count"] / panel["total_nsqip_cases"]

    # ARD merge
    ard_df["Most Recent RUC Review"] = pd.to_datetime(
        ard_df["Most Recent RUC Review"], errors="coerce"
    )
    ard_df["revaluation_year"] = ard_df["Most Recent RUC Review"].dt.year

    panel = panel.merge(
        ard_df[
            ["CPT", "Work RVU", "Intra Time", "Total Time",
             "Global", "Surgery? 0 =in office, 1 = surgery",
             "IWPUT", "2022 Medicare Utilization",
             "Top_Specialty", "revaluation_year"]
        ],
        on="CPT",
        how="left"
    )

    WINDOW = 5
    panel = panel[
        (panel["PUFYEAR"] >= panel["revaluation_year"] - WINDOW) &
        (panel["PUFYEAR"] <= panel["revaluation_year"] + WINDOW)
    ]

    panel["TREATED"] = (panel["PUFYEAR"] >= panel["revaluation_year"]).astype(int)

    return panel


# =========================================================
# VOLUME CAUSAL MODEL
# =========================================================
def run_volume_causal_forest(panel):

    panel = panel.copy()

    panel["Top_Specialty_enc"] = LabelEncoder().fit_transform(
        panel["Top_Specialty"].fillna("Unknown")
    )
    panel["Global_enc"] = LabelEncoder().fit_transform(
        panel["Global"].fillna("Unknown")
    )

    features = [
        "Work RVU", "Intra Time", "Total Time",
        "Surgery? 0 =in office, 1 = surgery",
        "IWPUT", "2022 Medicare Utilization",
        "Top_Specialty_enc", "Global_enc"
    ]

    panel = panel.dropna(subset=features + ["solo_share", "TREATED"])

    print("\n=== VOLUME PANEL ===")
    print(panel.shape)
    print(panel["TREATED"].value_counts())

    X = panel[features].values
    y = panel["solo_share"].values
    w = panel["TREATED"].values

    est = LinearDML(
        model_y=GradientBoostingRegressor(),
        model_t=GradientBoostingClassifier(),
        discrete_treatment=True,
        random_state=42
    )

    est.fit(Y=y, T=w, X=X)

    return est


# =========================================================
# MAIN (VOLUME ONLY)
# =========================================================
def main():

    nsqip_df = pd.read_csv("data/nsqip/combined_filtered.csv", low_memory=False)
    ard_df = pd.read_csv("data/final_CPT_1.csv")

    other_cols = [c for c in nsqip_df.columns if c.startswith("OTHERCPT")]
    solo_df = nsqip_df[nsqip_df[other_cols].isnull().all(axis=1)]

    vol_panel = build_volume_panel(
        solo_df,
        ard_df,
        nsqip_dir="data/nsqip"
    )

    print("\n=== SAMPLE VOLUME PANEL ===")
    print(vol_panel[[
        "CPT", "PUFYEAR",
        "filtered_count",
        "total_nsqip_cases",
        "solo_share"
    ]].head(10))

    run_volume_causal_forest(vol_panel)


if __name__ == "__main__":
    main()