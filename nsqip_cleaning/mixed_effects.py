import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from nsqip_segmented_lr import (
    load_ent_codes, load_data_for_reval, detect_revaluations_from_data,
    YEAR_START, YEAR_END
)

def load_total_cases(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df['YEAR'] = pd.to_numeric(df['PUFYEAR'], errors='coerce')
    df = df[(df['YEAR'] >= YEAR_START) & (df['YEAR'] <= YEAR_END)]
    return df.groupby('YEAR').size().reset_index(name='total_cases')


def build_panel_dataset(filepath, ent_codes, total_per_year, reval_map, direction_map, magnitude_map):
    #one row per CPT per year

    df = pd.read_csv(filepath, low_memory=False)
    df['YEAR'] = pd.to_numeric(df['PUFYEAR'], errors='coerce')
    df['CPT_NUM'] = pd.to_numeric(df['CPT'], errors='coerce')
    df_ent = df[df['CPT_NUM'].isin(ent_codes)].copy()
    df_ent = df_ent.dropna(subset=['YEAR'])
    df_ent = df_ent[(df_ent['YEAR'] >= YEAR_START) & (df_ent['YEAR'] <= YEAR_END)]
    
    yearly = df_ent.groupby(['CPT_NUM', 'YEAR']).size().reset_index(name='count')
    yearly = yearly.merge(total_per_year, on='YEAR')
    yearly['volume_rate'] = yearly['count'] / yearly['total_cases'] * 100
    
    rows = []
    for cpt_num in yearly['CPT_NUM'].unique():
        cpt_str = str(int(cpt_num))
        if cpt_str not in reval_map:
            continue
        
        reval_years = reval_map[cpt_str]
        best_year = max(reval_years, key=lambda y: magnitude_map.get(cpt_str, {}).get(y, 0))
        direction = direction_map.get(cpt_str, {}).get(best_year, 'unknown')
        magnitude = magnitude_map.get(cpt_str, {}).get(best_year, 0)
        
        cpt_data = yearly[yearly['CPT_NUM'] == cpt_num].sort_values('YEAR')
        
        for _, row in cpt_data.iterrows():
            year = int(row['YEAR'])
            rows.append({
                'cpt': cpt_str,
                'year': year,
                'reval_year': best_year,
                'year_c': year - best_year,
                'post': 1 if year >= best_year else 0,
                'volume_rate': row['volume_rate'],
                'reval_magnitude': magnitude,
                'reval_direction': direction,
                'is_decrease': 1 if direction == 'decrease' else 0,
            })
    
    return pd.DataFrame(rows)


def fit_linear_model(panel_df):
    """
    inear mixed-effects model with direction interaction.
    Random intercept per CPT.
    """
    print("LINEAR MIXED-EFFECTS MODEL -> Direction Interaction: DECREASES vs INCREASES")
    
    model = smf.mixedlm(
        "volume_rate ~ year_c + post + year_c:post + is_decrease + "
        "post:is_decrease + year_c:post:is_decrease",
        data=panel_df,
        groups=panel_df['cpt'],
        re_formula="1"
    )
    
    fit = model.fit(method='lbfgs', maxiter=1000)
    print(fit.summary())
    
    return fit

def interpret_results(fit, panel_df):
    print("KEY FINDINGS")
    
    # increases (reference group)
    post_inc = fit.params.get('post')
    post_inc_p = fit.pvalues.get('post')
    slope_inc = fit.params.get('year_c:post')
    slope_inc_p = fit.pvalues.get('year_c:post')
    
    # additional effect for decreases
    post_dec_diff = fit.params.get('post:is_decrease')
    post_dec_diff_p = fit.pvalues.get('post:is_decrease')
    slope_dec_diff = fit.params.get('year_c:post:is_decrease')
    slope_dec_diff_p = fit.pvalues.get('year_c:post:is_decrease')
    
    # total effect for decreases
    total_post_dec = post_inc + post_dec_diff
    total_slope_dec = slope_inc + slope_dec_diff
    
    n_inc = len(panel_df[panel_df['is_decrease'] == 0]['cpt'].unique())
    n_dec = len(panel_df[panel_df['is_decrease'] == 1]['cpt'].unique())
    
    print(f"\n  Dataset: {panel_df['cpt'].nunique()} CPTs ({n_inc} increases, {n_dec} decreases)")
    print(f"  Observations: {len(panel_df)}")
    print(f"  Log-Likelihood: {fit.llf:.1f}")
    
    print(f"\n  ── INCREASES (n={n_inc}) ──")
    print(f"  Immediate level change: {post_inc:+.4f} pp (p={post_inc_p:.4f})")
    print(f"  Slope change:           {slope_inc:+.4f} pp/yr (p={slope_inc_p:.4f})")
    
    print(f"\n  ── DECREASES (n={n_dec}) — difference from increases ──")
    print(f"  Level change difference: {post_dec_diff:+.4f} pp (p={post_dec_diff_p:.4f})")
    print(f"  Slope change difference: {slope_dec_diff:+.4f} pp/yr (p={slope_dec_diff_p:.4f})")
    
    print(f"\n  ── DECREASES (total effect) ──")
    direction = "drop" if total_post_dec < 0 else "increase"
    print(f"  Immediate level change: {total_post_dec:+.4f} pp ({direction})")
    trend_dir = "accelerated" if total_slope_dec > 0 else "decelerated"
    print(f"  Slope change:           {total_slope_dec:+.4f} pp/yr ({trend_dir})")
    
    print(f"\n  ── INTERPRETATION ──")
    if post_dec_diff_p < 0.05:
        print(f"  * Devaluations produce a significantly different immediate volume response")
        print(f"    than increases ({post_dec_diff:+.3f} pp difference, p={post_dec_diff_p:.4f})")
    if slope_dec_diff_p < 0.05:
        print(f"  * Devaluations produce a significantly different slope change")
        print(f"    than increases ({slope_dec_diff:+.3f} pp/yr difference, p={slope_dec_diff_p:.4f})")
    
    print(f"\n  Model: volume_rate ~ year_c + post + year_c:post + is_decrease +")
    print(f"          post:is_decrease + year_c:post:is_decrease + (1 | cpt)")


def main():
    print("MIXED-EFFECTS LINEAR MODEL")
    print("Volume Response to wRVU Revaluation")
    
    # load data + detect revaluations
    ent_codes = load_ent_codes('nsqip_cleaning/ENT_CPT_CODES.csv')
    print("\nLoading NSQIP Adult data")
    df_volume = load_data_for_reval('nsqip_cleaning/combined_filtered.csv', ent_codes)
    total_per_year = load_total_cases('nsqip_cleaning/combined_filtered.csv')
    
    reval_map, direction_map, magnitude_map = detect_revaluations_from_data(df_volume)
    
    # Build panel
    panel_df = build_panel_dataset(
        'nsqip_cleaning/combined_filtered.csv', ent_codes, total_per_year,
        reval_map, direction_map, magnitude_map
    )
    
    print(f"Panel: {len(panel_df)} rows, {panel_df['cpt'].nunique()} CPTs")
    
    # Fit model
    fit = fit_linear_model(panel_df)    
    interpret_results(fit, panel_df)
    
    panel_df.to_csv('mixed_linear_panel.csv', index=False)

if __name__ == "__main__":
    main()