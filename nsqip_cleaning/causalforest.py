import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from data_handling_nsqip import standardize_cpt


def clean_age(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace('+', '', regex=False), errors='coerce')


def build_case_level_panel(nsqip_df: pd.DataFrame, ard_df: pd.DataFrame) -> pd.DataFrame:
    nsqip_df = nsqip_df.copy()
    nsqip_df['CPT'] = standardize_cpt(nsqip_df['CPT'])
    ard_df['CPT'] = standardize_cpt(ard_df['CPT'])

    # revaluation year from ARD
    ard_df['Most Recent RUC Review'] = pd.to_datetime(
        ard_df['Most Recent RUC Review'], errors='coerce'
    )
    ard_df['revaluation_year'] = ard_df['Most Recent RUC Review'].dt.year

    ard_features = [
        'CPT', 'Work RVU', 'Intra Time', 'Total Time',
        'Global', 'Surgery? 0 =in office, 1 = surgery',
        'IWPUT', '2022 Medicare Utilization', 'Top_Specialty',
        'revaluation_year'
    ]

    # merge ARD features onto every individual case
    panel = nsqip_df.merge(ard_df[ard_features], on='CPT', how='inner')

    # treatment: was this case performed after revaluation?
    WINDOW = 5
    panel = panel[
        (panel['PUFYEAR'] >= panel['revaluation_year'] - WINDOW) &
        (panel['PUFYEAR'] <= panel['revaluation_year'] + WINDOW)
    ]
    panel['TREATED'] = (panel['PUFYEAR'] >= panel['revaluation_year']).astype(int)

    panel = panel.dropna(subset=['OPTIME', 'TREATED', 'Work RVU'])

    return panel


def plot_feature_importance(est, feature_cols: list, outcome_label: str = 'Operative Time'):
    """
    Bar chart of causal forest feature importances.
    """
    importances = est.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_features[::-1], sorted_importances[::-1], color='steelblue', alpha=0.7)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title(f'Causal Forest Feature Importances\nOutcome: {outcome_label}', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('figs/causal_forest_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_causal_forest(panel: pd.DataFrame, outcome: str = 'OPTIME'):  # changed default
    le = LabelEncoder()
    panel = panel.copy()
    panel['AGE'] = clean_age(panel['AGE'])
    panel['Top_Specialty_enc'] = le.fit_transform(panel['Top_Specialty'].fillna('Unknown'))
    panel['Global_enc'] = LabelEncoder().fit_transform(panel['Global'].fillna('Unknown'))

    feature_cols = [
        'Work RVU', 'Intra Time', 'Total Time',
        'Surgery? 0 =in office, 1 = surgery',
        'IWPUT', '2022 Medicare Utilization',
        'Top_Specialty_enc', 'Global_enc',
        'AGE',
    ]

    panel = panel.dropna(subset=feature_cols + [outcome, 'TREATED'])

    X = panel[feature_cols].values
    y = panel[outcome].values
    w = panel['TREATED'].values

    est = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
        model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3),
        discrete_treatment=True,
        n_estimators=2000,
        min_samples_leaf=5,
        random_state=42,
        inference=True
    )
    est.fit(Y=y, T=w, X=X)

    return est, feature_cols, panel, X


def plot_treatment_effects(est, X, panel: pd.DataFrame, outcome_label: str = 'Operative Time'):
    te = est.effect(X)
    te_lower, te_upper = est.effect_interval(X, alpha=0.05)

    panel = panel.copy()
    panel['TE'] = te
    panel['TE_lower'] = te_lower
    panel['TE_upper'] = te_upper

    # average TE per CPT for plotting
    cpt_te = (
        panel.groupby('CPT')[['TE', 'TE_lower', 'TE_upper']]
        .mean()
        .sort_values('TE')
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        cpt_te['CPT'], cpt_te['TE'],
        xerr=[
            cpt_te['TE'] - cpt_te['TE_lower'],
            cpt_te['TE_upper'] - cpt_te['TE']
        ],
        color=['firebrick' if x < 0 else 'steelblue' for x in cpt_te['TE']],
        alpha=0.7, capsize=3
    )
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel(f'Estimated Effect on {outcome_label} (minutes)', fontsize=11)
    ax.set_title(
        'Causal Forest: Estimated Treatment Effect of wRVU Revaluation\nper CPT Code (95% CI)',
        fontsize=12
    )
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('figs/causal_forest_treatment_effects.png', dpi=300, bbox_inches='tight')
    plt.show()

    return cpt_te


def main():
    nsqip_df = pd.read_csv("data/nsqip/combined_filtered.csv", low_memory=False)
    ard_df = pd.read_csv("data/final_CPT_1.csv")

    other_cols = [c for c in nsqip_df.columns if c.startswith('OTHERCPT')]
    nsqip_df = nsqip_df[nsqip_df[other_cols].isnull().all(axis=1)]

    panel = build_case_level_panel(nsqip_df, ard_df)
    print(f"Panel shape: {panel.shape}")
    print(f"Treated rows: {panel['TREATED'].sum()}, Control rows: {(panel['TREATED']==0).sum()}")
    print(f"Columns available: {panel.columns.tolist()}")  # sanity check

    est, feature_cols, panel, X = run_causal_forest(panel, outcome='OPTIME')

    plot_feature_importance(est, feature_cols, outcome_label='Operative Time')
    cpt_te = plot_treatment_effects(est, X, panel, outcome_label='Operative Time')
    cpt_te.to_csv('causal_forest_cpt_effects.csv', index=False)


if __name__ == "__main__":
    main()