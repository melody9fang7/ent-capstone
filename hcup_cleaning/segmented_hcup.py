import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from hcup_stats import group_map
from filtering import standardize_cpt
import warnings
warnings.filterwarnings('ignore')

MIN_RVU_CHANGE_PCT = 0.05
YEAR_START = 2008
YEAR_END = 2017
FOR_SINA = True

def load_reference_times(filepath="filtered_sina2.csv"):
    """
    Load reference intraoperative times from CSV.
    Returns dict: {cpt_str: intra_time_minutes}
    """
    ref = pd.read_csv(filepath)
    ref["CPT"] = standardize_cpt(ref["CPT Code"])
    
    ref_times = {}
    for _, row in ref.iterrows():
        cpt = row['CPT']
        intra_time = pd.to_numeric(row['Intra Time'], errors='coerce')
        if pd.notna(intra_time):
            ref_times[cpt] = intra_time
    
    print(f"Loaded reference times for {len(ref_times)} CPTs")
    return ref_times

REFERENCE_TIMES = load_reference_times("filtered_sina2.csv")

def get_ortime_data(df, cpt):
    """Get operative time data - ALL available years"""
    data = df[df['CPT1'] == cpt].copy()
    if len(data) < 10:
        return None
    return data[['AYEAR', 'ORTIME']].rename(columns={'AYEAR': 'YEAR', 'ORTIME': 'VALUE'})

def load_ortime_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded operative time data: {len(df):,} rows")

    df["ORTIME"] = pd.to_numeric(df["ORTIME"], errors="coerce")
    df["AYEAR"] = pd.to_numeric(df["AYEAR"], errors="coerce")
    df["CPT1"] = standardize_cpt(df["CPT1"])
    df = df.dropna(subset=["ORTIME", "AYEAR"])
    df = df[df["ORTIME"] > 0]
    df = df[(df["AYEAR"] >= YEAR_START) & (df["AYEAR"] <= YEAR_END)]

    return df

# use these funcs only on nsqip data
def extract_yearly_wrvu(cpt_file, nsqip_file, output_file, year_start = 2005, year_end = 2022, chunk_size = 150000):
    """
    Extracts yearly wRVUs from NSQIP dataset for CPT codes.
    """
    cpt_df = pd.read_csv(cpt_file)
    cpt_df["CPT"] = standardize_cpt(cpt_df["CPT1"])
    keep_cpts = set(cpt_df["CPT"])

    yearly_wrvu = {}

    print("Starting yearly wRVU extraction...")
    for chunk in pd.read_csv(nsqip_file, chunksize = chunk_size, low_memory = False):
        chunk["CPT"] = standardize_cpt(chunk["CPT"])
        chunk["PUFYEAR"] = pd.to_numeric(chunk["PUFYEAR"], errors = "coerce").astype("Int64")
        chunk["WORKRVU"] = pd.to_numeric(chunk["WORKRVU"], errors = "coerce")

        for i in range(1, 11):
            chunk[f"OTHERCPT{i}"] = standardize_cpt(chunk[f"OTHERCPT{i}"])
            chunk[f"OTHERWRVU{i}"] = pd.to_numeric(chunk[f"OTHERWRVU{i}"], errors = "coerce")

        main_match = chunk[chunk["CPT"].isin(keep_cpts)]
        for _, row in main_match.iterrows():
            cpt = row["CPT"]
            year = row["PUFYEAR"]
            wrvu = row["WORKRVU"]
            
            if pd.notna(year) and pd.notna(wrvu):
                if year_start <= year <= year_end:
                    if (cpt, int(year)) not in yearly_wrvu:
                        yearly_wrvu[(cpt, int(year))] = wrvu

        for i in range(1, 11):
            cpt_col = f"OTHERCPT{i}"
            wrvu_col = f"OTHERWRVU{i}"

            other_match = chunk[chunk[cpt_col].isin(keep_cpts)]
            for _, row in other_match.iterrows():
                cpt = row[cpt_col]
                year = row["PUFYEAR"]
                wrvu = row[wrvu_col]

                if pd.notna(year) and pd.notna(wrvu):
                    if year_start <= year <= year_end:
                        if (cpt, int(year)) not in yearly_wrvu:
                            yearly_wrvu[(cpt, int(year))] = wrvu

    yearly_df = pd.DataFrame({"CPT": cpt, "YEAR": year, "WORKRVU": wrvu} for (cpt, year), wrvu in yearly_wrvu.items())
    yearly_df = yearly_df.sort_values(["CPT", "YEAR"])
    manual_row = pd.DataFrame([{"CPT": "30520", "YEAR": 2007, "WORKRVU": 6.85}])
    yearly_df = pd.concat([yearly_df, manual_row], ignore_index = True)
    yearly_df = yearly_df.sort_values(["CPT", "YEAR"])
    yearly_df.to_csv(output_file, index = False)

    print("\nYearly wRVU extraction DONE.")
    print(f"Rows: {len(yearly_df)}")
    print(f"CPTs found: {yearly_df['CPT'].nunique()}")

    return yearly_df

def detect_revaluations_from_data(df_ent, min_change_pct=MIN_RVU_CHANGE_PCT):
    """
    Detect revaluation years AND direction from data.
    Parameters:
        df_ent: DataFrame with CPT_NUM, YEAR, WORKRVU columns
        min_change_pct: minimum cumulative percent change to flag as revaluation
    
    Returns:
        reval_map: {cpt: [years]}
        direction_map: {cpt: {year: 'increase' or 'decrease'}}
        magnitude_map: {cpt: {year: percent_change}}
    """
    #yearly_rvu = df_ent.groupby(['CPT_NUM', 'YEAR'])['WORKRVU'].mean().reset_index()
    yearly_rvu = df_ent.sort_values(['CPT', 'YEAR'])
    
    reval_map = {}
    direction_map = {}
    magnitude_map = {}
    
    for cpt in yearly_rvu['CPT'].unique():
        cpt_data = yearly_rvu[yearly_rvu['CPT'] == cpt].sort_values('YEAR').copy()
        
        if len(cpt_data) <= 1:
            continue
        
        change_years = []
        change_directions = {}
        change_magnitudes = {}
        
        # Establish baseline from first available year
        baseline_rvu = cpt_data.iloc[0]['WORKRVU']
        baseline_year = int(cpt_data.iloc[0]['YEAR'])
        prev_rvu = baseline_rvu
        
        # Track the last confirmed revaluation RVU separately
        last_reval_rvu = baseline_rvu
        last_reval_year = baseline_year
        
        for _, row in cpt_data.iterrows():
            current_rvu = row['WORKRVU']
            current_year = int(row['YEAR'])
            
            # Skip the baseline year itself
            if current_year == baseline_year:
                continue
            
            # Calculate change from LAST REVALUATION EVENT (not from previous year)
            if last_reval_rvu > 0:
                pct_change_from_reval = ((current_rvu - last_reval_rvu) / last_reval_rvu) * 100
            else:
                prev_rvu = current_rvu
                continue
            
            if abs(pct_change_from_reval) >= min_change_pct and (current_year - last_reval_year) >= 1:
                change_years.append(current_year)
                direction = 'increase' if current_rvu > last_reval_rvu else 'decrease'
                change_directions[current_year] = direction
                change_magnitudes[current_year] = abs(pct_change_from_reval)
                
                # Update revaluation baseline to this new level
                last_reval_rvu = current_rvu
                last_reval_year = current_year
            
            prev_rvu = current_rvu
        
        if change_years:
            # Ensure consistent string key for matching with other dataframes
            reval_map[str(int(cpt))] = change_years
            direction_map[str(int(cpt))] = change_directions
            magnitude_map[str(int(cpt))] = change_magnitudes
    
    return reval_map, direction_map, magnitude_map

# continuing on

def get_revaluation_info(cpt, year, direction_map, magnitude_map):
    """Get direction and magnitude for a specific revaluation event"""
    direction = direction_map.get(cpt, {}).get(year, None)
    magnitude = magnitude_map.get(cpt, {}).get(year, None)
    return direction, magnitude

def get_line_color(cpt, year, direction_map):
    """Return color for revaluation line: green for increase, red for decrease"""
    direction = direction_map.get(cpt, {}).get(year, None)
    if direction == 'increase':
        return 'green'
    elif direction == 'decrease':
        return 'red'
    else:
        return 'gray'

# MODEL FITTING

def build_design_matrix(data, break_years, include_level=False):
    """
    Build X matrix for segmented regression.
    Centralized so fitting and prediction use identical logic.
    """
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
    """Predict using the SAME design matrix logic as fitting."""
    fake_data = pd.DataFrame({'YEAR': years_range})
    X_pred, _ = build_design_matrix(fake_data, break_years, include_level=include_level)
    # Ensure columns match model
    X_pred = X_pred[model.params.index]
    return model.predict(X_pred)


def get_prediction_ci(model, data, break_years, include_level=False, alpha=0.05):
    """Get mean CI bands for predictions."""
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

def evaluate_breakpoints_slope_only(data, cpt, break_years, outcome_col, outcome_name):
    if not break_years or len(data) < 10:
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


def evaluate_breakpoints_level_slope(data, cpt, break_years, outcome_col, outcome_name):
    if not break_years or len(data) < 10:
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

# PRINTING

def print_results_table(results_df, outcome_name, direction_map=None, magnitude_map=None):
    model_type = results_df['Model_Type'].iloc[0] if len(results_df) > 0 else ''
    print(f"\n{outcome_name.upper()} SEGMENTED REGRESSION RESULTS ({model_type}):\n")
    print(f"{'CPT':<8} {'Break Years (Direction)':<35} {'n':<6} {'F-test p':<10} {'Signif':<7} {'RSS Red%':<9} {'R² Simple':<10} {'R² Seg':<10}")
    print("-"*120)
    for _, row in results_df.iterrows():
        cpt = row['CPT']
        if direction_map and magnitude_map:
            parts = []
            for by in row['Break_Years']:
                d, m = get_revaluation_info(cpt, by, direction_map, magnitude_map)
                arrow = '↑' if d == 'increase' else '↓'
                parts.append(f"{by} {arrow}({m:.0f}%)")
            by_str = ", ".join(parts)
        else:
            by_str = str(row['Break_Years'])
        sig = '✓' if row['Breakpoints_Significant'] else '✗'
        print(f"{cpt:<8} {by_str:<35} {row['n']:<6} {row['F_Pvalue']:.4f}   {sig:<7} {row['RSS_Reduction_Pct']:.1f}%     {row['R2_Simple']:.4f}   {row['R2_Segmented']:.4f}")
    print(f"\nSignificant: {results_df['Breakpoints_Significant'].sum()}/{len(results_df)} ({results_df['Breakpoints_Significant'].sum()/len(results_df)*100:.1f}%)")

def print_detailed_results(results_df, direction_map, magnitude_map):
    sig_df = results_df[results_df['Breakpoints_Significant'] == True]
    if len(sig_df) == 0:
        return
    
    print("DETAILED RESULTS FOR SIGNIFICANT CPTS")
    
    for _, row in sig_df.iterrows():
        cpt = row['CPT']
        print(f"\nCPT {cpt} | n={row['n']}")
        
        for by in row['Break_Years']:
            direction, magnitude = get_revaluation_info(cpt, by, direction_map, magnitude_map)
            arrow = '↑' if direction == 'increase' else '↓' if direction == 'decrease' else '?'
            print(f"   Reval {by}: {arrow} {magnitude:.1f}% ({direction})")
        
        print(f"   R²: {row['R2_Simple']:.4f} → {row['R2_Segmented']:.4f} (+{row['R2_Segmented']-row['R2_Simple']:.4f})")
        print(f"   RSS Reduction: {row['RSS_Reduction_Pct']:.1f}%")
        print(f"   F-test p = {row['F_Pvalue']:.4f} {'✓' if row['Breakpoints_Significant'] else '✗'}")
        print(f"   Pre-slope: {row['Pre_Slope']:+.2f}")
        for by, change in row['Slope_Changes'].items():
            pval = row['Slope_Pvalues'].get(by, 1)
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            direction, _ = get_revaluation_info(cpt, by, direction_map, magnitude_map)
            arrow = '↑' if direction == 'increase' else '↓' if direction == 'decrease' else '?'
            print(f"   → {by} {arrow}: slope change = {change:+.2f} {sig} (p={pval:.4f})")


# MODEL COMPARISON
def compare_models(slope_results, level_results):
    print("\nMODEL COMPARISON: Slope Only vs Level + Slope")
    slope_df = pd.DataFrame(slope_results)
    level_df = pd.DataFrame(level_results)
    print(f"{'Metric':<30} {'Slope Only':<15} {'Level + Slope':<15}")
    print(f"{'CPTs analyzed':<30} {len(slope_df):<15} {len(level_df):<15}")
    print(f"{'Significant (F-test)':<30} {slope_df['Breakpoints_Significant'].sum():<15} {level_df['Breakpoints_Significant'].sum():<15}")
    print(f"{'Mean R² (simple)':<30} {slope_df['R2_Simple'].mean():.4f}         {level_df['R2_Simple'].mean():.4f}")
    print(f"{'Mean R² (segmented)':<30} {slope_df['R2_Segmented'].mean():.4f}         {level_df['R2_Segmented'].mean():.4f}")
    print(f"{'Mean RSS Reduction %':<30} {slope_df['RSS_Reduction_Pct'].mean():.1f}%            {level_df['RSS_Reduction_Pct'].mean():.1f}%")
    print(f"\nMean AIC (Slope Only):     {slope_df['AIC'].mean():.1f}")
    print(f"Mean AIC (Level+Slope):    {level_df['AIC'].mean():.1f}")
    print(f"Mean BIC (Slope Only):     {slope_df['BIC'].mean():.1f}")
    print(f"Mean BIC (Level+Slope):    {level_df['BIC'].mean():.1f}")
    print(f"CPTs where Level+Slope BIC < Slope Only BIC: {(level_df['BIC'] < slope_df['BIC']).sum()}/{len(slope_df)}")
    print(f"CPTs where Level+Slope AIC < Slope Only AIC: {(level_df['AIC'] < slope_df['AIC']).sum()}/{len(slope_df)}\n")
    print(f"Mean RMSE (Slope Only):      {slope_df['RMSE'].mean():.2f}")
    print(f"Mean RMSE (Level+Slope):     {level_df['RMSE'].mean():.2f}")
    print(f"Mean MAE (Slope Only):       {slope_df['MAE'].mean():.2f}")
    print(f"Mean MAE (Level+Slope):      {level_df['MAE'].mean():.2f}")
    print(f"Mean Adj R² (Slope Only):    {slope_df['Adj_R2'].mean():.4f}")
    print(f"Mean Adj R² (Level+Slope):   {level_df['Adj_R2'].mean():.4f}")
    if len(slope_df) == len(level_df):
        better = (level_df['R2_Segmented'].values > slope_df['R2_Segmented'].values).sum()
        print(f"\nLevel+Slope R² > Slope Only R²: {better}/{len(slope_df)} CPTs")
    if 'Any_Level_Sig' in level_df.columns:
        print(f"CPTs with significant level change: {level_df['Any_Level_Sig'].sum()}/{len(level_df)}")
    if 'Any_Slope_Sig' in level_df.columns:
        print(f"CPTs with significant slope change: {level_df['Any_Slope_Sig'].sum()}/{len(level_df)}")

# DIAGNOSTICS


def regression_diagnostics(model, data, cpt, model_name, output_dir='diagnostics'):
    os.makedirs(output_dir, exist_ok=True)
    fitted = model.predict()
    residuals = model.resid
    dw = durbin_watson(residuals)
    try:
        bp = het_breuschpagan(residuals, model.model.exog)
        bp_pval = bp[1]
    except:
        bp_pval = np.nan
    sw_pval = stats.shapiro(residuals)[1] if len(residuals) >= 3 else np.nan
    print(f"\n  Diagnostics — CPT {cpt} ({model_name}):")
    print(f"    DW: {dw:.3f} {'✓' if 1.5 < dw < 2.5 else '!'}")
    print(f"    BP p: {bp_pval:.4f} {'✓' if bp_pval > 0.05 else '!'}")
    print(f"    SW p: {sw_pval:.4f} {'✓' if sw_pval > 0.05 else '!'}")
    return {'dw': dw, 'bp_pval': bp_pval, 'sw_pval': sw_pval, 'cpt': cpt, 'model': model_name}


def run_diagnostics_all_cpts(ortime_data_dict, reval_map, model_type='slope_only'):
    results = []
    for cpt, data in ortime_data_dict.items():
        break_years = reval_map.get(cpt, [])
        if not break_years:
            continue
        try:
            if model_type == 'slope_only':
                model, _, _ = fit_segmented_slope_only(data, break_years, 'VALUE')
            else:
                model, _, _, _ = fit_segmented_level_slope(data, break_years, 'VALUE')
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


def multiple_testing_correction(results_df):
    pvals = results_df['F_Pvalue'].dropna().values
    if len(pvals) == 0:
        return results_df
    reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
    results_df = results_df.copy()
    results_df['F_Pvalue_FDR'] = np.nan
    results_df.loc[results_df['F_Pvalue'].notna(), 'F_Pvalue_FDR'] = pvals_corrected
    results_df['Significant_FDR'] = results_df['F_Pvalue_FDR'] < 0.05
    print(f"\nMultiple Testing Correction (FDR):")
    print(f"  Significant (uncorrected): {results_df['Breakpoints_Significant'].sum()}/{len(results_df)}")
    print(f"  Significant (FDR corrected): {results_df['Significant_FDR'].sum()}/{len(results_df)}")
    return results_df

# PLOTTING

CPT_GROUPS = {
    "21556": "Head & Neck",
    "30520": "Septoplasty",
    "42415": "Head & Neck",
    "42440": "Head & Neck",
    "60220": "Head & Neck",
    "60240": "Head & Neck",
}

def plot_specific_cpts_single_model(data_dict, results_df, reval_map, direction_map,
                                     outcome_name, ylabel, filename, cpt_list,
                                     model_type='slope_only', show_ci=True):
    """
    Plot selected CPTs
    Set model_type='slope_only' for segreg with only slope, otherwise set to something else for level + slope
    """
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
        ref_time = REFERENCE_TIMES.get(cpt, None)
        if data is None or len(data) < 6:
            continue
        
        yearly_means = data.groupby('YEAR')['VALUE'].mean()
        break_years = reval_map.get(cpt, [])
        include_level = (model_type == 'level_slope')
        
        # Observed
        ax.plot(yearly_means.index, yearly_means.values, 'o', color='steelblue',
               alpha=0.8, markersize=10, zorder=3)
        
        # Model fit
        try:
            if model_type == 'slope_only':
                model, _, _ = fit_segmented_slope_only(data, break_years, 'VALUE')
            else:
                model, _, _, _ = fit_segmented_level_slope(data, break_years, 'VALUE')
            
            years_range = np.arange(int(data['YEAR'].min()), int(data['YEAR'].max()) + 1)
            pred = predict_from_model(model, break_years, years_range, include_level=include_level)
            
            label = 'Slope-Only Fit' if model_type == 'slope_only' else 'Level+Slope Fit'
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
        
        # Reference line
        if ref_time is not None:
            ax.axhline(y=ref_time, color='#C59E01', linestyle='--', linewidth=3, alpha=0.7)
        
        # Breakpoints
        for by in break_years:
            ax.axvline(x=by, color=get_line_color(cpt, by, direction_map),
                      linestyle='--', linewidth=3, alpha=0.7, zorder=1)
        
        # Y-axis
        y_data = yearly_means.values
        y_range = y_data.max() - y_data.min()
        if y_range < 5:
            y_center = (y_data.max() + y_data.min()) / 2
            y_min, y_max = y_center - 2.5, y_center + 2.5
        else:
            pad = y_range * 0.15
            y_min, y_max = y_data.min() - pad, y_data.max() + pad
        if ref_time is not None:
            y_min = min(y_min, ref_time - 1)
            y_max = max(y_max, ref_time + 1)
        ax.set_ylim(max(0, y_min), y_max)
        
        # Stats
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
        tick_step = 2
        ax.set_xticks(range(x_min, x_max + 1, tick_step))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    from matplotlib.lines import Line2D
    fit_label = 'Slope-Only Fit' if model_type == 'slope_only' else 'Level+Slope Fit'
    legend_handles = [
        Line2D([0], [0], color='steelblue', marker='o', markersize=10, linewidth=0, label='Observed Mean'),
        Line2D([0], [0], color='#c0392b', linewidth=3, label=fit_label),
        Line2D([0], [0], color='#C59E01', linestyle='--', linewidth=3, label='RUC Reference Time'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=3, label='wRVU Increase'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=3, label='wRVU Decrease'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=14,
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
    print("HCUP SEGMENTED REGRESSION ANALYSIS")
    cpt_list = 'hcup_cpt_counts.csv'
    nsqip_file = 'combined_filtered_930.csv'
    ref_file = "filtered_sina2.csv"

    # Load data
    df_ortime = load_ortime_data('HCUP_filtered_172_cleaned.csv')
    df_volume = extract_yearly_wrvu(cpt_list, nsqip_file, output_file = 'yearly_wrvu.csv')
    reval_map, direction_map, magnitude_map = detect_revaluations_from_data(df_volume)

    # keep only selected cpts
    keep_cpts = set(group_map.keys())
    filtered_reval_map = {
        cpt: years
        for cpt, years in reval_map.items()
        if cpt in keep_cpts
    }

    print(f"\nOriginal CPTs with detected revals: {len(filtered_reval_map)}")

    # to get right range for hcup (2008, 2017)
    filtered_reval_map = {
        cpt: [y for y in years if YEAR_START <= y <= YEAR_END]
        for cpt, years in filtered_reval_map.items()
        if any(YEAR_START <= y <= YEAR_END for y in years)
    }

    print(f"CPTs with breakpoints inside {YEAR_START}-{YEAR_END}: {len(filtered_reval_map)}")
    print(f"Found {len(filtered_reval_map)} CPTs with revaluations")
    
    print("OPERATIVE TIME ANALYSIS")    
    ortime_data_dict = {}
    
    for cpt, break_years in filtered_reval_map.items():
        if cpt not in df_ortime['CPT1'].unique():
            continue
        data = get_ortime_data(df_ortime, cpt)
        if data is not None:
            ortime_data_dict[cpt] = data
    
    # Fit both models
    slope_results, level_results = [], []
    for cpt, break_years in filtered_reval_map.items():
        if cpt not in ortime_data_dict:
            continue
        data = ortime_data_dict[cpt]
        r_s = evaluate_breakpoints_slope_only(data, cpt, break_years, 'VALUE', 'Operative Time')
        if r_s:
            slope_results.append(r_s)
        r_l = evaluate_breakpoints_level_slope(data, cpt, break_years, 'VALUE', 'Operative Time')
        if r_l:
            level_results.append(r_l)

    slope_df = pd.DataFrame(slope_results)
    level_df = pd.DataFrame(level_results)
    
    # FDR
    slope_df = multiple_testing_correction(slope_df)
    level_df = multiple_testing_correction(level_df)
    
    # Print
    print_results_table(slope_df, "Operative Time (Slope Only)", direction_map, magnitude_map)
    print_results_table(level_df, "Operative Time (Level + Slope)", direction_map, magnitude_map)
    
    # Compare
    compare_models(slope_results, level_results)
    
    # Diagnostics — all CPTs
    print("DIAGNOSTICS — ALL CPTs")
    for name, df in [('Slope Only', run_diagnostics_all_cpts(ortime_data_dict, filtered_reval_map, 'slope_only')),
                      ('Level+Slope', run_diagnostics_all_cpts(ortime_data_dict, filtered_reval_map, 'level_slope'))]:
        print(f"\n{name} (n={len(df)} CPTs):")
        print(f"  DW: Mean {df['DW'].mean():.2f} (range: {df['DW'].min():.2f}–{df['DW'].max():.2f})")
        print(f"    Outside [1.5,2.5]: {(~df['DW_OK']).sum()}/{len(df)}")
        print(f"  BP significant: {df['BP_sig'].sum()}/{len(df)}")
        print(f"  SW significant: {df['SW_sig'].sum()}/{len(df)}")

    # plotting 

    target_cpts = ['21556', '30520', '38542', '42415', '42420', '42440', '60220', '60240']

    plot_specific_cpts_single_model(
        ortime_data_dict, slope_df, filtered_reval_map, direction_map,
        "Operative Time Response", "Operative Time (minutes)",
        "segmented_ortime_slope_only.svg", target_cpts,
        model_type='slope_only', show_ci=True)
    
    plot_specific_cpts_single_model(
        ortime_data_dict, level_df, filtered_reval_map, direction_map,
        "Operative Time Response", "Operative Time (minutes)",
        "segmented_ortime_level_slope.svg", target_cpts,
        model_type='level_slope', show_ci=True)
    
    slope_df.to_csv('ortime_slope_only_results.csv', index=False)
    level_df.to_csv('ortime_level_slope_results.csv', index=False)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
