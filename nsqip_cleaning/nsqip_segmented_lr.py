import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

MIN_RVU_CHANGE_PCT = 0.05
YEAR_START = 2005
YEAR_END = 2022

def load_ent_codes(filepath):
    ent_cpt_df = pd.read_csv(filepath)
    first_col = ent_cpt_df.iloc[:, 0]
    ent_cpt_codes = set()
    for val in first_col:
        try:
            if pd.notna(val):
                val_str = str(val).strip()
                if val_str.replace('.', '').replace('-', '').isdigit():
                    cpt_int = int(float(val_str))
                    ent_cpt_codes.add(cpt_int)
        except:
            continue
    print(f"Loaded {len(ent_cpt_codes)} ENT CPT codes")
    return ent_cpt_codes

def load_optime_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded operative time data: {len(df):,} rows")
    df['OPTIME'] = pd.to_numeric(df['OPTIME'], errors='coerce')
    df['YEAR'] = pd.to_numeric(df['PUFYEAR'], errors='coerce')
    df['CPT'] = df['CPT'].astype(str)
    df = df.dropna(subset=['OPTIME', 'YEAR'])
    return df

def load_volume_data(filepath, ent_codes):
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df):,} rows")
    df['YEAR'] = pd.to_numeric(df['PUFYEAR'], errors='coerce')
    df['CPT_NUM'] = pd.to_numeric(df['CPT'], errors='coerce')
    df['WORKRVU'] = pd.to_numeric(df['WORKRVU'], errors='coerce')
    
    df_ent = df[df['CPT_NUM'].isin(ent_codes)].copy()
    df_ent = df_ent.dropna(subset=['YEAR', 'WORKRVU'])
    df_ent = df_ent[(df_ent['YEAR'] >= YEAR_START) & (df_ent['YEAR'] <= YEAR_END)]
    print(f"ENT procedures: {len(df_ent):,} rows")
    return df_ent

# DYNAMIC REVALUATION DETECTION

def detect_revaluations_from_data(df_ent, min_change_pct=MIN_RVU_CHANGE_PCT):
    """
    Detect revaluation years AND direction from data
    Returns:
        reval_map: {cpt: [years]}
        direction_map: {cpt: {year: 'increase' or 'decrease'}}
        magnitude_map: {cpt: {year: percent_change}}
    """
    yearly_rvu = df_ent.groupby(['CPT_NUM', 'YEAR'])['WORKRVU'].mean().reset_index()
    yearly_rvu = yearly_rvu.sort_values(['CPT_NUM', 'YEAR'])
    
    reval_map = {}
    direction_map = {}
    magnitude_map = {}
    
    for cpt in yearly_rvu['CPT_NUM'].unique():
        cpt_data = yearly_rvu[yearly_rvu['CPT_NUM'] == cpt].copy()
        
        if len(cpt_data) <= 1:
            continue
        
        changes = []
        change_years = []
        change_directions = {}
        change_magnitudes = {}
        
        prev_rvu = cpt_data.iloc[0]['WORKRVU']
        
        for _, row in cpt_data.iterrows():
            current_rvu = row['WORKRVU']
            year = int(row['YEAR'])
            pct_change = (current_rvu - prev_rvu) / prev_rvu * 100
            
            if abs(pct_change) >= min_change_pct:
                change_years.append(year)
                direction = 'increase' if current_rvu > prev_rvu else 'decrease'
                change_directions[year] = direction
                change_magnitudes[year] = abs(pct_change)
                changes.append({'year': year, 'direction': direction, 'pct_change': pct_change})
                prev_rvu = current_rvu
            else:
                prev_rvu = current_rvu
        
        if change_years:
            reval_map[str(cpt)] = change_years
            direction_map[str(cpt)] = change_directions
            magnitude_map[str(cpt)] = change_magnitudes
    
    return reval_map, direction_map, magnitude_map

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


def fit_segmented(data, break_years, outcome_col):
    """Fit segmented regression with breakpoints ALL years"""
    data = data.sort_values('YEAR').copy()
    X = data[['YEAR']].copy()
    X['const'] = 1
    
    for by in break_years:
        data[f'TIME_SINCE_{by}'] = np.maximum(0, data['YEAR'] - by)
        X[f'TIME_SINCE_{by}'] = data[f'TIME_SINCE_{by}']
    
    model = sm.OLS(data[outcome_col], X).fit()
    
    slopes = [model.params['YEAR']]
    slope_changes = [model.params.get(f'TIME_SINCE_{by}', 0) for by in break_years]
    for sc in slope_changes:
        slopes.append(slopes[-1] + sc)
    
    return model, slopes, slope_changes

def evaluate_breakpoints(data, cpt, break_years, outcome_col, outcome_name):
    """Evaluate breakpoint significances ALL years"""
    if not break_years or len(data) < 10:
        return None
    
    X_simple = sm.add_constant(data['YEAR'])
    simple_model = sm.OLS(data[outcome_col], X_simple).fit()
    
    try:
        seg_model, slopes, slope_changes = fit_segmented(data, break_years, outcome_col)
    except:
        return None
    
    rss_simple = np.sum(simple_model.resid**2)
    rss_seg = np.sum(seg_model.resid**2)
    rss_reduction_pct = (rss_simple - rss_seg) / rss_simple * 100 if rss_simple > 0 else 0
    
    df_diff = len(seg_model.params) - len(simple_model.params)
    if df_diff > 0:
        f_stat = ((rss_simple - rss_seg) / df_diff) / (rss_seg / (len(data) - len(seg_model.params)))
        f_pvalue = 1 - stats.f.cdf(f_stat, df_diff, len(data) - len(seg_model.params))
    else:
        f_pvalue = np.nan
    
    slope_pvalues = {}
    for by in break_years:
        slope_pvalues[by] = seg_model.pvalues.get(f'TIME_SINCE_{by}', 1.0)
    
    return {
        'CPT': cpt, 'Break_Years': break_years, 'Outcome': outcome_name,
        'n': len(data), 'R2_Simple': simple_model.rsquared, 'R2_Segmented': seg_model.rsquared,
        'RSS_Reduction_Pct': rss_reduction_pct, 'F_Pvalue': f_pvalue,
        'Breakpoints_Significant': f_pvalue < 0.05,
        'Pre_Slope': slopes[0], 'Segment_Slopes': slopes[1:],
        'Slope_Changes': dict(zip(break_years, slope_changes)),
        'Slope_Pvalues': slope_pvalues
    }


def get_optime_data(df, cpt):
    """Get operative time data - ALL available years"""
    data = df[df['CPT'] == cpt].copy()
    if len(data) < 10:
        return None
    return data[['YEAR', 'OPTIME']].rename(columns={'OPTIME': 'VALUE'})


def print_results_table(results_df, outcome_name, direction_map=None, magnitude_map=None):
    """Print formatted results table with direction indicators"""
    print(f"\n{'='*120}")
    print(f"{outcome_name.upper()} RESULTS (Dynamically Detected Revaluations)")
    print(f"{'='*120}")
    print(f"{'CPT':<8} {'Break Years (Direction)':<35} {'n':<6} {'F-test p':<10} {'Signif':<7} {'RSS Red%':<9} {'R² Simple':<10} {'R² Seg':<10}")
    print("-"*120)
    
    for _, row in results_df.iterrows():
        cpt = row['CPT']
        break_years = row['Break_Years']
        
        if direction_map and magnitude_map:
            by_str_parts = []
            for by in break_years:
                direction, magnitude = get_revaluation_info(cpt, by, direction_map, magnitude_map)
                if direction == 'increase':
                    by_str_parts.append(f"{by} ↑({magnitude:.0f}%)")
                elif direction == 'decrease':
                    by_str_parts.append(f"{by} ↓({magnitude:.0f}%)")
                else:
                    by_str_parts.append(f"{by} (?)")
            by_str = ", ".join(by_str_parts)
        else:
            by_str = str(break_years)
        
        sig = '✓' if row['Breakpoints_Significant'] else '✗'
        
        print(f"{cpt:<8} {by_str:<35} {row['n']:<6} {row['F_Pvalue']:.4f}   {sig:<7} {row['RSS_Reduction_Pct']:.1f}%     {row['R2_Simple']:.4f}   {row['R2_Segmented']:.4f}")
    
    sig_count = results_df['Breakpoints_Significant'].sum()
    print(f"\nSignificant: {sig_count}/{len(results_df)} ({sig_count/len(results_df)*100:.1f}%)")

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


def plot_results(data_dict, results_df, reval_map, direction_map, outcome_name, ylabel, filename):
    """Create segmented regression plots with direction-based colors"""
    sig_cpts = results_df[results_df['Breakpoints_Significant'] == True]['CPT'].tolist()
    if not sig_cpts:
        sig_cpts = list(reval_map.keys())[:9]
    
    cpts_to_plot = sig_cpts[:9]
    n_cols, n_rows = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, cpt in enumerate(cpts_to_plot):
        ax = axes[idx]
        data = data_dict.get(cpt)
        if data is None or len(data) < 6:
            ax.text(0.5, 0.5, f'CPT {cpt}\nInsufficient data', ha='center', va='center')
            ax.set_title(f'CPT {cpt}')
            continue
        
        break_years = reval_map.get(cpt, [])
        yearly_means = data.groupby('YEAR')['VALUE'].mean()
        
        try:
            model, slopes, _ = fit_segmented(data, break_years, 'VALUE')
            years_range = np.arange(data['YEAR'].min(), data['YEAR'].max() + 1)
            X_pred = pd.DataFrame({'YEAR': years_range})
            X_pred['const'] = 1
            for by in break_years:
                X_pred[f'TIME_SINCE_{by}'] = np.maximum(0, years_range - by)
            predictions = model.predict(X_pred)
            
            ax.plot(yearly_means.index, yearly_means.values, 'o-', color='steelblue', 
                   alpha=0.7, markersize=4, linewidth=1.5, label='Observed')
            ax.plot(years_range, predictions, 'r-', linewidth=2, label='Segmented')
            
            for by in break_years:
                color = get_line_color(cpt, by, direction_map)
                ax.axvline(x=by, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
        except:
            ax.plot(yearly_means.index, yearly_means.values, 'o-', color='steelblue', alpha=0.7)
        
        row = results_df[results_df['CPT'] == cpt]
        if len(row) > 0:
            r2 = row.iloc[0]['R2_Segmented']
            f_p = row.iloc[0]['F_Pvalue']
            ax.text(0.98, 0.98, f'R²={r2:.3f}\np={f_p:.4f}', transform=ax.transAxes,
                   va='top', ha='right', fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.set_title(f'CPT {cpt} (n={len(data):,})')
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Segmented Regression: {outcome_name}\n(Green = wRVU Increase, Red = wRVU Decrease)', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

# MAIN

def main():
    print("SEGMENTED REGRESSION ANALYSIS")
    
    # Load data
    df_optime = load_optime_data('nsqip_cleaning/combined_filtered_29.csv')
    ent_codes = load_ent_codes('nsqip_cleaning/ENT_CPT_CODES.csv')
    df_volume = load_volume_data('nsqip_cleaning/combined_filtered.csv', ent_codes)
        
    reval_map, direction_map, magnitude_map = detect_revaluations_from_data(df_volume)
    
    print(f"Found {len(reval_map)} CPTs with revaluations (≥{MIN_RVU_CHANGE_PCT}% change):")
    for cpt, years in list(reval_map.items())[:15]:
        dir_strs = []
        for y in years:
            d, m = get_revaluation_info(cpt, y, direction_map, magnitude_map)
            arrow = '↑' if d == 'increase' else '↓' if d == 'decrease' else '?'
            dir_strs.append(f"{y}{arrow}({m:.0f}%)")
        print(f"  CPT {cpt}: {', '.join(dir_strs)}")
    
    print("OPERATIVE TIME ANALYSIS")    
    optime_results = []
    optime_data_dict = {}
    
    for cpt, break_years in reval_map.items():
        if cpt not in df_optime['CPT'].unique():
            continue
        
        data = get_optime_data(df_optime, cpt)
        if data is not None:
            optime_data_dict[cpt] = data
            result = evaluate_breakpoints(data, cpt, break_years, 'VALUE', 'Operative Time')
            if result:
                optime_results.append(result)
    
    optime_df = pd.DataFrame(optime_results)
    if len(optime_df) > 0:
        print_results_table(optime_df, "Operative Time", direction_map, magnitude_map)
        print_detailed_results(optime_df, direction_map, magnitude_map)
        plot_results(optime_data_dict, optime_df, reval_map, direction_map,
                    "Operative Time Response", "Operative Time (minutes)", "segmented_optime_dynamic.png")
        optime_df.to_csv('optime_segmented_results_dynamic.csv', index=False)
    
    # FINAL SUMMARY
    print("FINAL SUMMARY")
    
    inc_count = sum(1 for cpt, years in reval_map.items() 
                    for y in years if direction_map.get(cpt, {}).get(y) == 'increase')
    dec_count = sum(1 for cpt, years in reval_map.items() 
                    for y in years if direction_map.get(cpt, {}).get(y) == 'decrease')
    
    print(f"Total revaluation events detected: {inc_count + dec_count}")
    print(f"  wRVU INCREASES (↑): {inc_count}")
    print(f"  wRVU DECREASES (↓): {dec_count}")
    print(f"\nOperative Time: {len(optime_df[optime_df['Breakpoints_Significant']==True])}/{len(optime_df)} significant")
    print("\nSaved: optime_segmented_results_dynamic.csv")
    print("Saved: segmented_optime_dynamic.png")

if __name__ == "__main__":
    main()