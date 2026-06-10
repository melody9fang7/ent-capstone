import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from nsqip_segmented_lr import (
    load_ent_codes, load_data_for_reval, detect_revaluations_from_data,
    YEAR_START, YEAR_END
)

# 1. LOAD

import os

def load_total_cases(filepath, cache='total_cases_cache.csv'):
    """Load total cases per year, with caching."""
    if os.path.exists(cache):
        return pd.read_csv(cache)
    
    df = pd.read_csv(filepath, usecols=['PUFYEAR'], low_memory=False)
    df['YEAR'] = pd.to_numeric(df['PUFYEAR'], errors='coerce')
    df = df[(df['YEAR'] >= YEAR_START) & (df['YEAR'] <= YEAR_END)]
    result = df.groupby('YEAR').size().reset_index(name='total_cases')
    result.to_csv(cache, index=False)
    return result


def build_panel(filepath, ent_codes, total_per_year, reval_map, direction_map, magnitude_map):
    """One row per CPT per year."""
    df = pd.read_csv(filepath, low_memory=False)
    df['YEAR'] = pd.to_numeric(df['PUFYEAR'], errors='coerce')
    df['CPT_NUM'] = pd.to_numeric(df['CPT'], errors='coerce')
    df_ent = df[df['CPT_NUM'].isin(ent_codes)].dropna(subset=['YEAR'])
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
        direction = direction_map.get(cpt_str, {}).get(best_year, 'unknown')  # ← use direction_map
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


# 2. FIT MODEL


def fit_model(panel_df):
    """Fit mixed-effects model — try progressively simpler if singular."""
    
    # Model 1: Full
    try:
        model = smf.mixedlm(
            "volume_rate ~ year_c + post + year_c:post + post:is_decrease + year_c:post:is_decrease",
            data=panel_df,
            groups=panel_df['cpt'],
            re_formula="1"
        )
        fit = model.fit(method='lbfgs', maxiter=1000)
        print("Model: Full (with 3-way interaction)")
        print(fit.summary())
        return fit
    except:
        pass
    return fit


# 3. INTERPRET

def calculate_r2_mixed(fit, panel_df):
    """Calculate marginal and conditional R² for mixed model."""
    # Fixed effects predictions (population average)
    fixed_pred = fit.predict(exog=panel_df)
    
    # Full model predictions (with random effects)
    full_pred = fit.fittedvalues
    
    # Variance components
    var_fixed = np.var(fixed_pred)
    var_random = fit.cov_re.iloc[0, 0] if hasattr(fit.cov_re, 'iloc') else fit.cov_re['Group']
    var_residual = fit.scale
    var_total = var_fixed + var_random + var_residual
    
    marginal_r2 = var_fixed / var_total
    conditional_r2 = (var_fixed + var_random) / var_total
    
    print(f"\nModel Fit:")
    print(f"  Marginal R² (fixed effects only): {marginal_r2:.4f}")
    print(f"  Conditional R² (fixed + random):  {conditional_r2:.4f}")
    
    return marginal_r2, conditional_r2

def print_results(fit, panel_df):
    """Simple plain-language output."""
    
    post_inc = fit.params['post']
    slope_inc = fit.params['year_c:post']
    post_dec = fit.params['post:is_decrease']
    slope_dec = fit.params['year_c:post:is_decrease']
    
    n_inc = len(panel_df[panel_df['is_decrease'] == 0]['cpt'].unique())
    n_dec = len(panel_df[panel_df['is_decrease'] == 1]['cpt'].unique())
    
    print(f"MIXED-EFFECTS MODEL RESULTS")
    print(f"CPTs: {panel_df['cpt'].nunique()} ({n_inc} increases, {n_dec} decreases)")
    print(f"Observations: {len(panel_df)}")
    
    print(f"\n── INCREASES (n={n_inc}) ──")
    print(f"  Level change: {post_inc:+.4f} pp (p={fit.pvalues['post']:.4f})")
    print(f"  Slope change: {slope_inc:+.4f} pp/yr (p={fit.pvalues['year_c:post']:.4f})")
    
    print(f"\n── DECREASES (n={n_dec}) — difference from increases ──")
    print(f"  Level change diff: {post_dec:+.4f} pp (p={fit.pvalues['post:is_decrease']:.4f})")
    print(f"  Slope change diff: {slope_dec:+.4f} pp/yr (p={fit.pvalues['year_c:post:is_decrease']:.4f})")
    
    print(f"\n── DECREASES (total effect) ──")
    print(f"  Level change: {post_inc + post_dec:+.4f} pp")
    print(f"  Slope change: {slope_inc + slope_dec:+.4f} pp/yr")
    
    if fit.pvalues['post:is_decrease'] < 0.05:
        print(f"\n✓ Devaluations produce significantly different volume response than increases")
    if fit.pvalues['year_c:post:is_decrease'] < 0.05:
        print(f"✓ Devaluations produce significantly different slope change than increases")


# 4. PLOT

def plot_result(panel_df):
    """Simple direction comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'increase': '#2ecc71', 'decrease': '#e74c3c'}
    
    for direction in ['increase', 'decrease']:
        sub = panel_df[panel_df['is_decrease'] == (1 if direction == 'decrease' else 0)]
        avg = sub.groupby('year_c')['volume_rate'].agg(['mean', 'sem']).reset_index()
        
        ax.plot(avg['year_c'], avg['mean'], 'o-', color=colors[direction], 
               linewidth=2.5, markersize=8, label=f'{direction} (n={sub["cpt"].nunique()} CPTs)')
        ax.fill_between(avg['year_c'], avg['mean'] - 1.96*avg['sem'], 
                       avg['mean'] + 1.96*avg['sem'], color=colors[direction], alpha=0.15)
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Revaluation')
    ax.set_xlabel('Years from Revaluation', fontsize=14)
    ax.set_ylabel('Mean Volume (% of NSQIP Cases)', fontsize=14)
    ax.set_title('Volume Response by Revaluation Direction', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('mixed_direction_simple.svg', facecolor='white')
    plt.show()

def plot_diagnostics(fit, panel_df):
    """Residuals vs Fitted + Q-Q plot for mixed model."""
    fitted = fit.fittedvalues
    residuals = fit.resid
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ── Residuals vs Fitted ──
    ax = axes[0]
    for direction, color in [('increase', '#2ecc71'), ('decrease', '#e74c3c')]:
        is_dec = 1 if direction == 'decrease' else 0
        mask = panel_df['is_decrease'] == is_dec
        ax.scatter(fitted[mask], residuals[mask], alpha=0.4, s=15, 
                  color=color, label=f'{direction} (n={mask.sum()})')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Fitted Values', fontsize=13)
    ax.set_ylabel('Residuals', fontsize=13)
    ax.set_title('Residuals vs Fitted', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ── Q-Q Plot ──
    ax = axes[1]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor('steelblue')
    ax.get_lines()[0].set_markeredgecolor('white')
    ax.get_lines()[0].set_alpha(0.4)
    ax.get_lines()[1].set_color('red')
    ax.set_title('Q-Q Plot — Residuals', fontsize=14, fontweight='bold')
    
    plt.suptitle('Mixed-Effects Model Diagnostics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mixed_diagnostics.svg', facecolor='white', bbox_inches='tight')
    plt.show()
    
    # ── Print summary ──
    group_var = fit.cov_re.iloc[0, 0] if hasattr(fit.cov_re, 'iloc') else fit.cov_re['Group']
    icc = group_var / (group_var + fit.scale)
    print(f"\nDiagnostics:")
    print(f"  Model converged: ✓")
    print(f"  Groups: {panel_df['cpt'].nunique()} CPTs")
    print(f"  ICC: {icc:.2f} ({icc*100:.0f}% of variance between CPTs)")



# 5. MAIN

def main():
    # Load
    ent_codes = load_ent_codes('nsqip_cleaning/ENT_CPT_CODES.csv')
    df_volume = load_data_for_reval('nsqip_cleaning/combined_filtered.csv', ent_codes)
    total_per_year = load_total_cases('nsqip_cleaning/combined_filtered.csv')
    reval_map, direction_map, magnitude_map = detect_revaluations_from_data(df_volume)
    
    # Build panel
    panel_df = build_panel('nsqip_cleaning/combined_filtered.csv', ent_codes, total_per_year, reval_map, direction_map, magnitude_map)
    print(f"Panel: {len(panel_df)} rows, {panel_df['cpt'].nunique()} CPTs")
    
    # Fit
    fit = fit_model(panel_df)
    print(fit.summary())
    
    # Diagnostics
    plot_diagnostics(fit, panel_df)
    marg_r2, cond_r2 = calculate_r2_mixed(fit, panel_df)

    # Results
    print_results(fit, panel_df)
    
    # Plot
    plot_result(panel_df)


if __name__ == "__main__":
    main()
