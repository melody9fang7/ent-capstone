import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
import os
import warnings
warnings.filterwarnings('ignore')
# For each CPT code:
# Standardized Volume = (Count of that CPT in that year / Total NSQIP cases that year) × 100

from nsqip_segmented_lr import (
    load_ent_codes, load_data_for_reval, detect_revaluations_from_data,
    get_revaluation_info, get_line_color,
    YEAR_START, YEAR_END, MIN_RVU_CHANGE_PCT
)

CPT_GROUPS = {
    '38542': 'Neck Dissection',
    '42415': 'Salivary Gland',
    '42420': 'Salivary Gland',
    '42440': 'Salivary Gland',
    '60220': 'Thyroid',
    '60240': 'Thyroid',
}

# LOADING

def load_total_cases_per_year(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df['YEAR'] = pd.to_numeric(df['PUFYEAR'], errors='coerce')
    df = df[(df['YEAR'] >= YEAR_START) & (df['YEAR'] <= YEAR_END)]
    total_per_year = df.groupby('YEAR').size().reset_index(name='total_cases')
    print(f"Total cases per year:\n{total_per_year.to_string(index=False)}")
    return total_per_year


def build_volume_data(filepath, ent_codes, total_per_year):
    df = pd.read_csv(filepath, low_memory=False)
    df['YEAR'] = pd.to_numeric(df['PUFYEAR'], errors='coerce')
    df['CPT_NUM'] = pd.to_numeric(df['CPT'], errors='coerce')
    df_ent = df[df['CPT_NUM'].isin(ent_codes)].copy()
    df_ent = df_ent.dropna(subset=['YEAR'])
    df_ent = df_ent[(df_ent['YEAR'] >= YEAR_START) & (df_ent['YEAR'] <= YEAR_END)]
    yearly_counts = df_ent.groupby(['CPT_NUM', 'YEAR']).size().reset_index(name='count')
    yearly_counts = yearly_counts.merge(total_per_year, on='YEAR')
    yearly_counts['VALUE'] = yearly_counts['count'] / yearly_counts['total_cases'] * 100
    volume_data_dict = {}
    for cpt_num in yearly_counts['CPT_NUM'].unique():
        cpt_str = str(int(cpt_num))
        cpt_data = yearly_counts[yearly_counts['CPT_NUM'] == cpt_num][['YEAR', 'VALUE']].copy()
        if len(cpt_data) >= 5:
            volume_data_dict[cpt_str] = cpt_data
    print(f"Built volume data for {len(volume_data_dict)} CPTs")
    return volume_data_dict


# MODEL FITTING

def build_design_matrix(data, break_years, include_level=False):
    data = data.sort_values('YEAR').copy()
    X = pd.DataFrame({'YEAR': data['YEAR'].values, 'const': 1.0})
    for by in sorted(break_years):
        X[f'TIME_SINCE_{by}'] = np.maximum(0, data['YEAR'].values - by)
        if include_level:
            X[f'POST_{by}'] = (data['YEAR'].values >= by).astype(float)
    return X, data


def fit_segmented_slope_only(data, break_years, outcome_col):
    X, data = build_design_matrix(data, break_years, include_level=False)
    model = sm.OLS(data[outcome_col].values, X).fit(cov_type='HC3')
    slope_changes = {by: model.params.get(f'TIME_SINCE_{by}', 0) for by in break_years}
    slopes = [model.params['YEAR']]
    for by in sorted(break_years):
        slopes.append(slopes[-1] + slope_changes[by])
    return model, slopes, slope_changes


def fit_segmented_level_slope(data, break_years, outcome_col):
    X, data = build_design_matrix(data, break_years, include_level=True)
    model = sm.OLS(data[outcome_col].values, X).fit(cov_type='HC3')
    level_changes = {by: model.params.get(f'POST_{by}', 0) for by in break_years}
    slope_changes = {by: model.params.get(f'TIME_SINCE_{by}', 0) for by in break_years}
    slopes = [model.params['YEAR']]
    for by in sorted(break_years):
        slopes.append(slopes[-1] + slope_changes[by])
    return model, slopes, level_changes, slope_changes


def predict_from_model(model, break_years, years_range, include_level=False):
    fake_data = pd.DataFrame({'YEAR': years_range})
    X_pred, _ = build_design_matrix(fake_data, break_years, include_level=include_level)
    X_pred = X_pred[model.params.index]
    return model.predict(X_pred)


def get_prediction_ci(model, data, break_years, include_level=False, alpha=0.05):
    years_range = np.arange(int(data['YEAR'].min()), int(data['YEAR'].max()) + 1)
    fake_data = pd.DataFrame({'YEAR': years_range})
    X_pred, _ = build_design_matrix(fake_data, break_years, include_level=include_level)
    X_pred = X_pred[model.params.index]
    predictions = model.get_prediction(X_pred)
    pred_summary = predictions.summary_frame(alpha=alpha)
    return {
        'years': years_range,
        'fitted': pred_summary['mean'].values,
        'ci_lower': pred_summary['mean_ci_lower'].values,
        'ci_upper': pred_summary['mean_ci_upper'].values,
    }


# EVALUATION

def evaluate_volume_slope_only(data, cpt, break_years, outcome_col, outcome_name):
    if not break_years or len(data) < 6:
        return None
    X_simple = sm.add_constant(data['YEAR'])
    simple_model = sm.OLS(data[outcome_col], X_simple).fit(cov_type='HC3')
    try:
        seg_model, slopes, slope_changes = fit_segmented_slope_only(data, break_years, outcome_col)
    except:
        return None
    rss_simple = np.sum(simple_model.resid**2)
    rss_seg = np.sum(seg_model.resid**2)
    rss_red = (rss_simple - rss_seg) / rss_simple * 100 if rss_simple > 0 else 0
    df_diff = len(seg_model.params) - len(simple_model.params)
    f_pvalue = np.nan
    if df_diff > 0:
        f_stat = ((rss_simple - rss_seg) / df_diff) / (rss_seg / (len(data) - len(seg_model.params)))
        f_pvalue = 1 - stats.f.cdf(f_stat, df_diff, len(data) - len(seg_model.params))
    slope_pvalues = {by: seg_model.pvalues.get(f'TIME_SINCE_{by}', 1.0) for by in break_years}
    return {
        'CPT': cpt, 'Break_Years': break_years, 'Outcome': outcome_name,
        'n': len(data), 'R2_Simple': simple_model.rsquared, 'R2_Segmented': seg_model.rsquared,
        'RSS_Reduction_Pct': rss_red, 'F_Pvalue': f_pvalue,
        'Breakpoints_Significant': f_pvalue < 0.05,
        'Pre_Slope': slopes[0], 'Segment_Slopes': slopes[1:],
        'Slope_Changes': slope_changes, 'Slope_Pvalues': slope_pvalues,
        'Model_Type': 'Slope Only',
        'AIC': seg_model.aic, 'BIC': seg_model.bic,
        'RMSE': np.sqrt(mean_squared_error(data[outcome_col], seg_model.predict())),
        'MAE': mean_absolute_error(data[outcome_col], seg_model.predict()),
        'Adj_R2': 1 - (1 - seg_model.rsquared) * (len(data) - 1) / (len(data) - len(seg_model.params)),
    }


def evaluate_volume_level_slope(data, cpt, break_years, outcome_col, outcome_name):
    if not break_years or len(data) < 6:
        return None
    X_simple = sm.add_constant(data['YEAR'])
    simple_model = sm.OLS(data[outcome_col], X_simple).fit(cov_type='HC3')
    try:
        seg_model, slopes, level_changes, slope_changes = fit_segmented_level_slope(data, break_years, outcome_col)
    except:
        return None
    rss_simple = np.sum(simple_model.resid**2)
    rss_seg = np.sum(seg_model.resid**2)
    rss_red = (rss_simple - rss_seg) / rss_simple * 100 if rss_simple > 0 else 0
    df_diff = len(seg_model.params) - len(simple_model.params)
    f_pvalue = np.nan
    if df_diff > 0:
        f_stat = ((rss_simple - rss_seg) / df_diff) / (rss_seg / (len(data) - len(seg_model.params)))
        f_pvalue = 1 - stats.f.cdf(f_stat, df_diff, len(data) - len(seg_model.params))
    level_pvalues = {by: seg_model.pvalues.get(f'POST_{by}', 1.0) for by in break_years}
    slope_pvalues = {by: seg_model.pvalues.get(f'TIME_SINCE_{by}', 1.0) for by in break_years}
    return {
        'CPT': cpt, 'Break_Years': break_years, 'Outcome': outcome_name,
        'n': len(data), 'R2_Simple': simple_model.rsquared, 'R2_Segmented': seg_model.rsquared,
        'RSS_Reduction_Pct': rss_red, 'F_Pvalue': f_pvalue,
        'Breakpoints_Significant': f_pvalue < 0.05,
        'Pre_Slope': slopes[0], 'Segment_Slopes': slopes[1:],
        'Level_Changes': level_changes, 'Slope_Changes': slope_changes,
        'Level_Pvalues': level_pvalues, 'Slope_Pvalues': slope_pvalues,
        'Any_Level_Sig': any(p < 0.05 for p in level_pvalues.values()),
        'Any_Slope_Sig': any(p < 0.05 for p in slope_pvalues.values()),
        'Model_Type': 'Level + Slope',
        'AIC': seg_model.aic, 'BIC': seg_model.bic,
        'RMSE': np.sqrt(mean_squared_error(data[outcome_col], seg_model.predict())),
        'MAE': mean_absolute_error(data[outcome_col], seg_model.predict()),
        'Adj_R2': 1 - (1 - seg_model.rsquared) * (len(data) - 1) / (len(data) - len(seg_model.params)),
    }


# PRINTING and MODEL COMPARISON

def print_volume_table(results_df, outcome_name):
    model_type = results_df['Model_Type'].iloc[0] if len(results_df) > 0 else ''
    print(f"\n{outcome_name.upper()} ({model_type}):")
    print(f"{'CPT':<8} {'F-test p':<10} {'Sig':<5} {'RSS Red%':<10} {'R² Simple':<10} {'R² Seg':<10} {'AIC':<10}")
    for _, row in results_df.iterrows():
        sig = '✓' if row['Breakpoints_Significant'] else '✗'
        print(f"{row['CPT']:<8} {row['F_Pvalue']:.4f}     {sig:<5} {row['RSS_Reduction_Pct']:.1f}%       {row['R2_Simple']:.4f}     {row['R2_Segmented']:.4f} {row['AIC']:.1f}")
    print(f"\nSignificant: {results_df['Breakpoints_Significant'].sum()}/{len(results_df)}")


def compare_models(slope_results, level_results):
    print("\n" + "="*80)
    print("MODEL COMPARISON: Slope Only vs Level + Slope")
    print("="*80)
    slope_df = pd.DataFrame(slope_results)
    level_df = pd.DataFrame(level_results)
    if len(slope_df) == 0 or len(level_df) == 0:
        print("Insufficient results")
        return
    print(f"{'Metric':<30} {'Slope Only':<15} {'Level + Slope':<15}")
    print(f"{'CPTs analyzed':<30} {len(slope_df):<15} {len(level_df):<15}")
    print(f"{'Significant (F-test)':<30} {slope_df['Breakpoints_Significant'].sum():<15} {level_df['Breakpoints_Significant'].sum():<15}")
    print(f"{'Mean R² (simple)':<30} {slope_df['R2_Simple'].mean():.4f}         {level_df['R2_Simple'].mean():.4f}")
    print(f"{'Mean R² (segmented)':<30} {slope_df['R2_Segmented'].mean():.4f}         {level_df['R2_Segmented'].mean():.4f}")
    print(f"{'Mean RSS Reduction %':<30} {slope_df['RSS_Reduction_Pct'].mean():.1f}%            {level_df['RSS_Reduction_Pct'].mean():.1f}%")
    print(f"Mean AIC (Slope Only):     {slope_df['AIC'].mean():.1f}")
    print(f"Mean AIC (Level+Slope):    {level_df['AIC'].mean():.1f}")
    print(f"Mean BIC (Slope Only):     {slope_df['BIC'].mean():.1f}")
    print(f"Mean BIC (Level+Slope):    {level_df['BIC'].mean():.1f}")
    aic_better = (level_df['AIC'] < slope_df['AIC']).sum()
    bic_better = (level_df['BIC'] < slope_df['BIC']).sum()
    print(f"CPTs where Level+Slope AIC < Slope Only AIC: {aic_better}/{len(slope_df)}")
    print(f"CPTs where Level+Slope BIC < Slope Only BIC: {bic_better}/{len(slope_df)}")
    print(f"Mean RMSE (Slope Only):      {slope_df['RMSE'].mean():.4f}")
    print(f"Mean RMSE (Level+Slope):     {level_df['RMSE'].mean():.4f}")
    print(f"Mean Adj R² (Slope Only):    {slope_df['Adj_R2'].mean():.4f}")
    print(f"Mean Adj R² (Level+Slope):   {level_df['Adj_R2'].mean():.4f}")
    if 'Any_Level_Sig' in level_df.columns:
        print(f"CPTs with significant level change: {level_df['Any_Level_Sig'].sum()}/{len(level_df)}")
    if 'Any_Slope_Sig' in level_df.columns:
        print(f"CPTs with significant slope change: {level_df['Any_Slope_Sig'].sum()}/{len(level_df)}")


def multiple_testing_correction(results_df):
    pvals = results_df['F_Pvalue'].dropna().values
    if len(pvals) == 0:
        return results_df
    reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
    results_df = results_df.copy()
    results_df['F_Pvalue_FDR'] = np.nan
    results_df.loc[results_df['F_Pvalue'].notna(), 'F_Pvalue_FDR'] = pvals_corrected
    results_df['Significant_FDR'] = results_df['F_Pvalue_FDR'] < 0.05
    print(f"\nFDR Correction: {results_df['Breakpoints_Significant'].sum()} → {results_df['Significant_FDR'].sum()}/{len(results_df)}")
    return results_df

# DIAGNOSTICS

def run_diagnostics_all_cpts(volume_data_dict, reval_map, model_type='slope_only'):
    results = []
    for cpt, data in volume_data_dict.items():
        break_years = reval_map.get(cpt, [])
        if not break_years:
            continue
        valid_breaks = [by for by in break_years if data['YEAR'].min() <= by <= data['YEAR'].max()]
        if not valid_breaks:
            continue
        try:
            if model_type == 'slope_only':
                model, _, _ = fit_segmented_slope_only(data, valid_breaks, 'VALUE')
            else:
                model, _, _, _ = fit_segmented_level_slope(data, valid_breaks, 'VALUE')
            residuals = model.resid
            dw = durbin_watson(residuals)
            try:
                bp = het_breuschpagan(residuals, model.model.exog)
                bp_pval = bp[1]
            except:
                bp_pval = np.nan
            sw_pval = stats.shapiro(residuals)[1] if len(residuals) >= 3 else np.nan
            results.append({
                'CPT': cpt, 'n': len(data),
                'DW': round(dw, 3), 'DW_OK': 1.5 < dw < 2.5,
                'BP_pval': round(bp_pval, 4) if not np.isnan(bp_pval) else np.nan,
                'BP_sig': bp_pval < 0.05 if not np.isnan(bp_pval) else np.nan,
                'SW_pval': round(sw_pval, 4) if not np.isnan(sw_pval) else np.nan,
                'SW_sig': sw_pval < 0.05 if not np.isnan(sw_pval) else np.nan,
            })
        except:
            continue
    return pd.DataFrame(results)

# PLOTTING

def plot_volume_single_model(data_dict, results_df, reval_map, direction_map,
                              outcome_name, ylabel, filename, cpt_list,
                              model_type='slope_only', show_ci=True):
    cpts_to_plot = [cpt for cpt in cpt_list if cpt in data_dict]
    if len(cpts_to_plot) == 0:
        return
    
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 13))
    axes = axes.flatten()
    
    for idx, cpt in enumerate(cpts_to_plot):
        ax = axes[idx]
        data = data_dict.get(cpt)
        group = CPT_GROUPS.get(cpt, '')
        if data is None or len(data) < 6:
            continue
        
        yearly_means = data.groupby('YEAR')['VALUE'].mean()
        break_years = reval_map.get(cpt, [])
        include_level = (model_type == 'level_slope')
        
        ax.plot(yearly_means.index, yearly_means.values, 'o', color='steelblue',
               alpha=0.8, markersize=10, zorder=3)
        
        try:
            if model_type == 'slope_only':
                model, _, _ = fit_segmented_slope_only(data, break_years, 'VALUE')
            else:
                model, _, _, _ = fit_segmented_level_slope(data, break_years, 'VALUE')
            
            years_range = np.arange(int(data['YEAR'].min()), int(data['YEAR'].max()) + 1)
            pred = predict_from_model(model, break_years, years_range, include_level=include_level)
            
            label = 'Slope-Only' if model_type == 'slope_only' else 'Level+Slope'
            ax.plot(years_range, pred, '-', color='#c0392b', linewidth=3, alpha=0.9,
                   label=label, zorder=4)
            
            if show_ci:
                try:
                    ci = get_prediction_ci(model, data, break_years, include_level=include_level)
                    ax.fill_between(ci['years'], ci['ci_lower'], ci['ci_upper'],
                                   color='#c0392b', alpha=0.1)
                except:
                    pass
        except:
            pass
        
        for by in break_years:
            ax.axvline(x=by, color=get_line_color(cpt, by, direction_map),
                      linestyle='--', linewidth=3, alpha=0.7, zorder=1)
        
        y_data = yearly_means.values
        y_range = y_data.max() - y_data.min()
        pad = max(y_range * 0.15, 0.001)
        ax.set_ylim(max(0, y_data.min() - pad), y_data.max() + pad)
        
        row = results_df[results_df['CPT'] == cpt]
        if len(row) > 0:
            f_p = row.iloc[0]['F_Pvalue']
            sig = '*' if row.iloc[0]['Breakpoints_Significant'] else ''
            ax.text(0.98, 0.96, f'p={f_p:.4f}{sig}', transform=ax.transAxes,
                   va='top', ha='right', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        ax.set_xlabel('Year', fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_title(f'CPT {cpt}: {group}', fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        x_min, x_max = int(yearly_means.index.min()), int(yearly_means.index.max())
        tick_step = max(1, (x_max - x_min) // 5)
        ax.set_xticks(range(x_min, x_max + 1, tick_step))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    from matplotlib.lines import Line2D
    fit_label = 'Slope-Only Fit' if model_type == 'slope_only' else 'Level+Slope Fit'
    legend_handles = [
        Line2D([0], [0], color='steelblue', marker='o', markersize=10, linewidth=0, label='Observed Mean'),
        Line2D([0], [0], color='#c0392b', linewidth=3, label=fit_label),
        Line2D([0], [0], color='green', linestyle='--', linewidth=3, label='wRVU Increase'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=3, label='wRVU Decrease'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=14,
              frameon=True, bbox_to_anchor=(0.5, -0.02))
    
    model_name = 'Slope Only' if model_type == 'slope_only' else 'Level + Slope'
    plt.suptitle(f'Segmented Regression ({model_name}): {outcome_name}',
                fontsize=22, fontweight='bold', y=1.01)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.12, hspace=0.35, wspace=0.25)
    plt.savefig(filename, dpi=300, facecolor='white', bbox_inches="tight", pad_inches=0.3)
    plt.show()
    print(f"Saved: {filename}")


# MAIN

def main():
    print("NSQIP VOLUME — SEGMENTED REGRESSION WITH MODEL COMPARISON")
    
    ent_codes = load_ent_codes('nsqip_cleaning/ENT_CPT_CODES.csv')
    df_volume = load_data_for_reval('combined_no_filter.csv', ent_codes)
    total_per_year = load_total_cases_per_year('combined_no_filter.csv')
    volume_data_dict = build_volume_data('combined_no_filter.csv', ent_codes, total_per_year)
    reval_map, direction_map, magnitude_map = detect_revaluations_from_data(df_volume)
    print(f"Found {len(reval_map)} CPTs with revaluations")
    
    # Fit both models
    slope_results, level_results = [], []
    for cpt, break_years in reval_map.items():
        if cpt not in volume_data_dict:
            continue
        data = volume_data_dict[cpt]
        valid_breaks = [by for by in break_years if data['YEAR'].min() <= by <= data['YEAR'].max()]
        if not valid_breaks:
            continue
        
        r_s = evaluate_volume_slope_only(data, cpt, valid_breaks, 'VALUE', 'Volume')
        if r_s:
            slope_results.append(r_s)
        r_l = evaluate_volume_level_slope(data, cpt, valid_breaks, 'VALUE', 'Volume')
        if r_l:
            level_results.append(r_l)
    
    slope_df = pd.DataFrame(slope_results)
    level_df = pd.DataFrame(level_results)
    
    # FDR
    slope_df = multiple_testing_correction(slope_df)
    level_df = multiple_testing_correction(level_df)
    
    # Print
    print_volume_table(slope_df, "Volume (Slope Only)")
    print_volume_table(level_df, "Volume (Level + Slope)")
    
    # Compare
    compare_models(slope_results, level_results)
    
    # Diagnostics
    print("DIAGNOSTICS — ALL CPTs")
    for name, df in [('Slope Only', run_diagnostics_all_cpts(volume_data_dict, reval_map, 'slope_only')),
                      ('Level+Slope', run_diagnostics_all_cpts(volume_data_dict, reval_map, 'level_slope'))]:
        if len(df) > 0:
            print(f"\n{name} (n={len(df)} CPTs):")
            print(f"  DW: Mean {df['DW'].mean():.2f} (range: {df['DW'].min():.2f}–{df['DW'].max():.2f})")
            print(f"    Outside 1.5–2.5: {(~df['DW_OK']).sum()}/{len(df)}")
            print(f"  BP significant: {df['BP_sig'].sum()}/{len(df)}")
            print(f"  SW significant: {df['SW_sig'].sum()}/{len(df)}")
    
    # Plots
    target_cpts = ['38542', '42415', '42420', '42440', '60220', '60240']
    
    plot_volume_single_model(
        volume_data_dict, slope_df, reval_map, direction_map,
        "Procedural Volume Response", "% of Total NSQIP Cases",
        "segmented_volume_slope_only.svg", target_cpts,
        model_type='slope_only', show_ci=True)
    
    plot_volume_single_model(
        volume_data_dict, level_df, reval_map, direction_map,
        "Procedural Volume Response", "% of Total NSQIP Cases",
        "segmented_volume_level_slope.svg", target_cpts,
        model_type='level_slope', show_ci=True)
    
    slope_df.to_csv('volume_slope_only_results.csv', index=False)
    level_df.to_csv('volume_level_slope_results.csv', index=False)

if __name__ == "__main__":
    main()
