import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# GO DOWN TO MAIN AND MAKE SURE YOU HAVE NSQIP FILES AND REFERENCE FILES STORED CORRECTLY, OR CHANGE THE PATHS ON YOUR OWN

FOR_SINA = True

MIN_RVU_CHANGE_PCT = 0.05
YEAR_START = 2005
YEAR_END = 2022

CPT_GROUPS = {
    '38542': 'Neck Dissection',
    '42415': 'Salivary Gland',
    '42420': 'Salivary Gland',
    '42440': 'Salivary Gland',
    '60220': 'Thyroid',
    '60240': 'Thyroid',
}

GROUP_COLORS = {
    'Neck Dissection': '#540d6e',
    'Salivary Gland': '#ffa600',
    'Thyroid': '#bc4c96',
}

def load_reference_times(filepath='nsqip_cleaning/reference_times.csv'):
    """
    Load reference intraoperative times from CSV.
    Returns dict: {cpt_str: intra_time_minutes}
    """
    ref = pd.read_csv(filepath)
    ref['CPT'] = ref['CPT'].astype(str).str.strip()
    
    ref_times = {}
    for _, row in ref.iterrows():
        cpt = row['CPT']
        intra_time = pd.to_numeric(row['Intra Time'], errors='coerce')
        if pd.notna(intra_time):
            ref_times[cpt] = intra_time
    
    print(f"Loaded reference times for {len(ref_times)} CPTs")
    return ref_times

REFERENCE_TIMES = load_reference_times('nsqip_cleaning/reference_times.csv')

def filter_solo_cases(df):
    other_cols = [c for c in df.columns if c.startswith('OTHERCPT')]
    if other_cols:
        is_solo = df[other_cols].isnull().all(axis=1)
        print(f"Kept {is_solo.sum():,} solo cases ({is_solo.sum()/len(df)*100:.1f}%)")
        return df[is_solo].copy()
    return df

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
    print(f"{len(df):,} rows before filtering for solo procedures")
    df = filter_solo_cases(df)
    print(f"{len(df):,} rows after filtering for solo procedures")
    df['OPTIME'] = pd.to_numeric(df['OPTIME'], errors='coerce')
    df['YEAR'] = pd.to_numeric(df['PUFYEAR'], errors='coerce')
    df['CPT'] = df['CPT'].astype(str)
    df = df.dropna(subset=['OPTIME', 'YEAR'])
    return df

def load_data_for_reval(filepath, ent_codes):
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df):,} rows for revaluation detection")
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
    Detect revaluation years AND direction from data.
    Parameters:
        df_ent: DataFrame with CPT_NUM, YEAR, WORKRVU columns
        min_change_pct: minimum cumulative percent change to flag as revaluation
    
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
        cpt_data = yearly_rvu[yearly_rvu['CPT_NUM'] == cpt].sort_values('YEAR').copy()
        
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
    print(f"{outcome_name.upper()} SEGMENTED REGRESSION RESULTS:\n")
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


def plot_results(data_dict, results_df, reval_map, direction_map, outcome_name, ylabel, filename, 
                 significant_only=False):
    if significant_only:
        cpts_to_plot = results_df[results_df['Breakpoints_Significant'] == True]['CPT'].tolist()
    else:
        cpts_to_plot = results_df['CPT'].tolist()
    
    if not cpts_to_plot:
        cpts_to_plot = list(reval_map.keys())[:9]
        
    n_plots = len(cpts_to_plot)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, cpt in enumerate(cpts_to_plot):
        ax = axes[idx]
        data = data_dict.get(cpt)
        ref_time = REFERENCE_TIMES.get(cpt, None)
        
        if data is None or len(data) < 6:
            ax.text(0.5, 0.5, f'CPT {cpt}\nInsufficient data', ha='center', va='center', fontsize=14)
            ax.set_title(f'CPT {cpt}', fontsize=16, fontweight='bold')
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
                   alpha=0.8, markersize=8, linewidth=2, label='Observed')
            ax.plot(years_range, predictions, '-', color='#c0392b', linewidth=2.5, 
                   alpha=0.9, label='Regression')
            
            for by in break_years:
                color = get_line_color(cpt, by, direction_map)
                ax.axvline(x=by, color=color, linestyle='--', linewidth=2, alpha=0.7)
        except:
            ax.plot(yearly_means.index, yearly_means.values, 'o-', color='steelblue', alpha=0.7)
        
        # ── Reference line ──
        if ref_time is not None:
            ax.axhline(y=ref_time, color='#C59E01', linestyle='--', linewidth=3, 
                      alpha=0.6, label=f'RUC ({ref_time} min)')
        
        # ── Y-axis scaling ──
        y_min_data = yearly_means.values.min()
        y_max_data = yearly_means.values.max()
        y_range = y_max_data - y_min_data
        
        if y_range < 5:
            y_center = (y_max_data + y_min_data) / 2
            y_min = y_center - 2.5
            y_max = y_center + 2.5
        else:
            padding = y_range * 0.15
            y_min = y_min_data - padding
            y_max = y_max_data + padding
        
        if ref_time is not None:
            y_min = min(y_min, ref_time - 1)
            y_max = max(y_max, ref_time + 1)
        
        y_min = max(0, y_min)
        ax.set_ylim(y_min, y_max)
        
        # ── Stats ──
        row = results_df[results_df['CPT'] == cpt]
        if len(row) > 0:
            f_p = row.iloc[0]['F_Pvalue']
            sig = '*' if row.iloc[0]['Breakpoints_Significant'] else ''
            if not FOR_SINA:
                ax.text(0.98, 0.98, f'F-test p={f_p:.4f}{sig}\nn={len(data):,}', 
                       transform=ax.transAxes, va='top', ha='right', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        ax.set_xlabel('Year', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(f'CPT {cpt} (n={len(data):,})', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        
        # Integer x-axis
        x_min, x_max = int(yearly_means.index.min()), int(yearly_means.index.max())
        tick_step = max(1, (x_max - x_min) // 4)
        ax.set_xticks(range(x_min, x_max + 1, tick_step))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Segmented Regression: {outcome_name}\n'
                f'(Green = wRVU Increase, Red = wRVU Decrease, Dotted = RUC Reference Time)',
                fontsize=18, fontweight='bold')
    plt.savefig(filename, dpi=200, facecolor='white', format='svg')
    plt.show()
    print(f"Saved: {filename}")

def plot_specific_cpts(data_dict, results_df, reval_map, direction_map, magnitude_map, 
                       outcome_name, ylabel, filename, cpt_list):
    """
    segmented regression plots for selected CPTs.
    3 columns x 2 rows
    """
    cpts_to_plot = [cpt for cpt in cpt_list if cpt in data_dict]
    
    if len(cpts_to_plot) == 0:
        print(f"None of the specified CPTs found in data: {cpt_list}")
        return
    
    n_cols = 3
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    axes = axes.flatten()
    
    # For CSV export — just the yearly means
    export_rows = []
    
    for idx, cpt in enumerate(cpts_to_plot):
        ax = axes[idx]
        data = data_dict.get(cpt)
        group = CPT_GROUPS.get(cpt, '')
        ref_time = REFERENCE_TIMES.get(cpt, None)
        
        if data is None or len(data) < 6:
            ax.text(0.5, 0.5, f'CPT {cpt}: {group}\nInsufficient data', 
                   ha='center', va='center', fontsize=18)
            ax.set_title(f'CPT {cpt}: {group}', fontsize=22, fontweight='bold')
            continue
        
        yearly_means = data.groupby('YEAR')['VALUE'].mean()
        break_years = reval_map.get(cpt, [])
        
        # Build export rows
        for year, val in yearly_means.items():
            export_rows.append({
                'CPT': cpt,
                'Group': group,
                'Year': int(year),
                'Yearly_Mean': round(val, 3),
                'RUC_Reference_Time': ref_time if ref_time else '',
            })
        
        # Plot observed data 
        ax.plot(yearly_means.index, yearly_means.values, 'o-', 
               color='steelblue', alpha=0.8, markersize=10, linewidth=3, 
               zorder=3)
        
        # Fit and plot segmented regression
        try:
            model, slopes, _ = fit_segmented(data, break_years, 'VALUE')
            years_range = np.arange(int(data['YEAR'].min()), int(data['YEAR'].max()) + 1)
            X_pred = pd.DataFrame({'YEAR': years_range})
            X_pred['const'] = 1
            for by in break_years:
                X_pred[f'TIME_SINCE_{by}'] = np.maximum(0, years_range - by)
            predictions = model.predict(X_pred)
            ax.plot(years_range, predictions, '-', color='#c0392b', 
                   linewidth=3, alpha=0.9, zorder=4)
        except:
            pass
        
        # Reference line
        if ref_time is not None:
            ax.axhline(y=ref_time, color='#C59E01', linestyle='--', linewidth=3, alpha=0.7)
        
        # Breakpoint lines
        for by in break_years:
            color = get_line_color(cpt, by, direction_map)
            ax.axvline(x=by, color=color, linestyle='--', linewidth=3, alpha=0.7, zorder=1)
        
        # Stats annotation
        row = results_df[results_df['CPT'] == cpt] if len(results_df) > 0 else None
        if row is not None and len(row) > 0:
            f_p = row.iloc[0]['F_Pvalue']
            sig = '*' if row.iloc[0]['Breakpoints_Significant'] else ''
            if not FOR_SINA:
                ax.text(0.98, 0.96, f'F-test p={f_p:.4f}{sig}\nn={len(data):,}', 
                       transform=ax.transAxes, va='top', ha='right', fontsize=14,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                edgecolor='gray', alpha=0.9))
        
        # Y-axis scaling
        y_min_data = yearly_means.values.min()
        y_max_data = yearly_means.values.max()
        y_range = y_max_data - y_min_data
        
        if y_range < 5:
            y_center = (y_max_data + y_min_data) / 2
            y_min = y_center - 2.5
            y_max = y_center + 2.5
        else:
            padding = y_range * 0.15
            y_min = y_min_data - padding
            y_max = y_max_data + padding
        
        if ref_time is not None:
            y_min = min(y_min, ref_time - 1)
            y_max = max(y_max, ref_time + 1)
        
        y_min = max(0, y_min)
        ax.set_ylim(y_min, y_max)
        
        # Formatting
        ax.set_xlabel('Year', fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_title(f'CPT {cpt}: {group}', fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        x_min, x_max = int(yearly_means.index.min()), int(yearly_means.index.max())
        tick_step = max(1, (x_max - x_min) // 5)
        ax.set_xticks(range(x_min, x_max + 1, tick_step))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Hide unused panels
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    # Legend at bottom
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='steelblue', marker='o', markersize=10, linewidth=3, 
               label='Observed Mean'),
        Line2D([0], [0], color='#c0392b', linewidth=3, label='Regression'),
        Line2D([0], [0], color='#C59E01', linestyle='--', linewidth=3, label='RUC Reference Time'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=3, label='wRVU Increase'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=3, label='wRVU Decrease'),
    ]
    
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, 
              fontsize=16, frameon=True, bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle(f'Segmented Regression: {outcome_name}',
                fontsize=24, fontweight='bold', y=1.01)
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.10, 
                        hspace=0.35, wspace=0.25)
    
    plt.savefig(filename, dpi=300, facecolor='white', format='svg', bbox_inches="tight", pad_inches=0.3)
    plt.show()
    
    # to CSV
    export_df = pd.DataFrame(export_rows)
    csv_filename = filename.replace('.svg', '_data.csv')
    export_df.to_csv(csv_filename, index=False)
    
    print(f"\nSaved: {filename}")
    print(f"Saved: {csv_filename}\n")


def plot_single_cpt_optime(optime_data_dict, cpt, results_df, reval_map, direction_map,
                             ylabel='Operative Time (minutes)', filename=None):
    """
    Plot a single CPT
    """
    if filename is None:
        filename = f'nsqip_single_optime_{cpt}.svg'
    
    if cpt not in optime_data_dict:
        print(f"CPT {cpt} not in optime data")
        return
    
    data = optime_data_dict[cpt].sort_values('YEAR')
    break_years = reval_map.get(cpt, [])
    group = CPT_GROUPS.get(cpt, '')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    #  Plot observed data 
    yearly_means = data.groupby('YEAR')['VALUE'].mean()
    ax.plot(yearly_means.index, yearly_means.values, 'o-', 
           color='steelblue', alpha=0.9, markersize=20, linewidth=5, 
           markerfacecolor='white', markeredgewidth=3,
           label='Yearly Mean Operative Time', zorder=3)
    
    #  Fit and plot segmented regression 
    try:
        from nsqip_segmented_lr import fit_segmented
        model, slopes, _ = fit_segmented(data, break_years, 'VALUE')
        years_range = np.arange(int(data['YEAR'].min()), int(data['YEAR'].max()) + 1)
        X_pred = pd.DataFrame({'YEAR': years_range})
        X_pred['const'] = 1
        for by in break_years:
            X_pred[f'TIME_SINCE_{by}'] = np.maximum(0, years_range - by)
        predictions = model.predict(X_pred)
        ax.plot(years_range, predictions, '-', color='#c0392b', 
               linewidth=5, alpha=0.9, label='Segmented Regression', zorder=4)
    except:
        pass
    
    # Reference line 
    ref_time = REFERENCE_TIMES.get(cpt, None)
    if ref_time is not None:
        ax.axhline(y=ref_time, color='#C59E01', linestyle='--', linewidth=5, 
                  alpha=0.8, label=f'RUC Intra Time ({ref_time} min)')
    
    #  Breakpoint lines 
    for by in break_years:
        direction = direction_map.get(cpt, {}).get(by, None)
        if direction == 'increase':
            color, label = 'green', 'wRVU Increase'
        elif direction == 'decrease':
            color, label = 'red', 'wRVU Decrease'
        else:
            color, label = 'gray', 'Revaluation'
        ax.axvline(x=by, color=color, linestyle='--', linewidth=5, alpha=0.8, 
                  label=label, zorder=1)
    
    # Stats annotation
    if results_df is not None:
        row = results_df[results_df['CPT'] == cpt]
        if len(row) > 0:
            f_p = row.iloc[0]['F_Pvalue']
            sig = '*' if row.iloc[0]['Breakpoints_Significant'] else ''
            r2 = row.iloc[0]['R2_Segmented']
            ax.text(0.96, 0.94, f'F-test p = {f_p:.4f}{sig}\nR²={r2:.4f}', 
                   transform=ax.transAxes, va='top', ha='right', fontsize=24,
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                            edgecolor='gray', linewidth=2, alpha=0.9))
    
    # Y-axis scaling
    y_min_data = yearly_means.values.min()
    y_max_data = yearly_means.values.max()
    y_range = y_max_data - y_min_data
    
    if y_range < 5:
        y_center = (y_max_data + y_min_data) / 2
        y_min = y_center - 5
        y_max = y_center + 5
    else:
        padding = y_range * 0.2
        y_min = y_min_data - padding
        y_max = y_max_data + padding
    
    if ref_time is not None:
        y_min = min(y_min, ref_time - 3)
        y_max = max(y_max, ref_time + 3)
    
    y_min = max(0, y_min)
    ax.set_ylim(y_min, y_max)
    
    #  Formatting 
    ax.set_xlabel('Year', fontsize=28, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=28, fontweight='bold')
    ax.set_title(f'CPT {cpt}: {group} Operative Time', fontsize=35, fontweight='bold', pad=15)
    ax.tick_params(labelsize=22, width=2, length=8)
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Integer x-axis
    x_min, x_max = int(yearly_means.index.min()), int(yearly_means.index.max())
    tick_step = max(1, (x_max - x_min) // 6)
    ax.set_xticks(range(x_min, x_max + 1, tick_step))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    plt.savefig(filename, dpi=300, facecolor='white', format='svg')
    plt.show()
    print(f"Saved: {filename}")

# MAIN

def main():
    print("SEGMENTED REGRESSION ANALYSIS")
    # ── Shared ENT codes ──
    ent_codes = load_ent_codes('nsqip_cleaning/ENT_CPT_CODES.csv')

    
    ################################################################################

    # ──── CHOOSE ONE: ────
    
    # ── OPTION A: NSQIP Adult only ──
    df_optime = load_optime_data('nsqip_cleaning/combined_filtered_29.csv')
    df_volume = load_data_for_reval('nsqip_cleaning/combined_filtered_29.csv', ent_codes)
    
    # ── OPTION B: NSQIP-P Pediatric only ──
    # df_optime = load_optime_data('nsqip-pediatrics/NSQIP-P_combined_filtered_solo.csv')
    # df_volume = load_data_for_reval('nsqip-pediatrics/ALL_NSQIP-P.csv', ent_codes)  # ← use full file
    
    # ── OPTION C: Combined Adult + Pediatric ──
    #optime_adult = load_optime_data('nsqip_cleaning/combined_filtered_29.csv')
    #volume_adult = load_data_for_reval('nsqip_cleaning/combined_filtered_29.csv', ent_codes)
    
    #optime_peds = load_optime_data('nsqip-pediatrics/NSQIP-P_combined_filtered_solo.csv')
    #volume_peds = load_data_for_reval('nsqip-pediatrics/ALL_NSQIP-P.csv', ent_codes)
    
    #df_optime = pd.concat([optime_adult, optime_peds], ignore_index=True)
    #df_volume = pd.concat([volume_adult, volume_peds], ignore_index=True)
    
    #print(f"\nCombined optime: {len(df_optime):,} rows")
    #print(f"Combined volume: {len(df_volume):,} rows")

    # ──── ──────────────────────────────────── ────
    
    # Everything below this is the same regardless of data source
        
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
        #plot_results(optime_data_dict, optime_df, reval_map, direction_map,
        #            "Operative Time Response", "Operative Time (minutes)", "segmented_optime_dynamic.svg")

        # All CPTs
        #plot_results(optime_data_dict, optime_df, reval_map, direction_map,
        #            "Operative Time Response", "Operative Time (minutes)", 
        #            "segmented_optime_all.svg", significant_only=False)

        # Significant only
        plot_results(optime_data_dict, optime_df, reval_map, direction_map,
                    "Operative Time Response", "Operative Time (minutes)", 
                    "segmented_optime_significant.svg", significant_only=True)

        optime_df.to_csv('optime_segmented_results_dynamic.csv', index=False)

    target_cpts = ['38542', '42415', '42420', '42440', '60220', '60240']
    plot_specific_cpts(
        optime_data_dict, optime_df, reval_map, direction_map, magnitude_map,
        "Operative Time Response", "Operative Time (minutes)", "segmented_optime_selected_cpts.svg",
        target_cpts
    )
    
    plot_single_cpt_optime(
        optime_data_dict, '42440', optime_df, reval_map, direction_map,
        ylabel='Operative Time (minutes)',
        filename='nsqip_single_optime_42440.svg'
    )

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
    print("Saved: segmented_optime_dynamic.svg")

if __name__ == "__main__":
    main()
