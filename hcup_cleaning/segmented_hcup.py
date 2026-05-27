import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
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

def load_optime_data(filepath):
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
    data = df[df['CPT1'] == cpt].copy()
    if len(data) < 10:
        return None
    return data[['AYEAR', 'ORTIME']].rename(columns={'AYEAR': 'YEAR', 'ORTIME': 'VALUE'})


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
    """Create segmented regression plots with direction-based colors and reference lines."""
    sig_cpts = results_df[results_df['Breakpoints_Significant'] == True]['CPT'].tolist()
    if not sig_cpts:
        sig_cpts = list(reval_map.keys())[:9]
    
    cpts_to_plot = sig_cpts
    n_plots = len(cpts_to_plot)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
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
        tick_step = 2
        ax.set_xticks(range(x_min, x_max + 1, tick_step))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Segmented Regression: {outcome_name}\n'
                f'(Green = wRVU Increase, Red = wRVU Decrease, Dotted = RUC Reference Time)',
                fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', format='svg')
    plt.show()
    print(f"Saved: {filename}")

def plot_specific_cpts_OLD(data_dict, results_df, reval_map, direction_map, magnitude_map, outcome_name, ylabel, filename, cpt_list):
    cpts_to_plot = [cpt for cpt in cpt_list if cpt in data_dict]
    
    if len(cpts_to_plot) == 0:
        print(f"None of the specified CPTs found in data: {cpt_list}")
        return
    
    n_plots = len(cpts_to_plot)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
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
        
        # Plot data FIRST
        ax.plot(yearly_means.index, yearly_means.values, 'o-', color='steelblue', 
               alpha=0.7, markersize=4, linewidth=1.5, label='Observed')
        
        # Fit and plot segmented regression
        try:
            model, slopes, _ = fit_segmented(data, break_years, 'VALUE')
            years_range = np.arange(data['YEAR'].min(), data['YEAR'].max() + 1)
            X_pred = pd.DataFrame({'YEAR': years_range})
            X_pred['const'] = 1
            for by in break_years:
                X_pred[f'TIME_SINCE_{by}'] = np.maximum(0, years_range - by)
            predictions = model.predict(X_pred)
            ax.plot(years_range, predictions, 'r-', linewidth=2, label='Segmented')
        except:
            pass
        
        # breakpoint lines and annotations (after data is plotted, so ylim is correct)
        y_min, y_max = ax.get_ylim()
        for i, by in enumerate(break_years):
            color = get_line_color(cpt, by, direction_map)
            ax.axvline(x=by, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
        
        row = results_df[results_df['CPT'] == cpt] if len(results_df) > 0 else None
        row = results_df[results_df['CPT'] == cpt]
        if len(row) > 0:
            f_p = row.iloc[0]['F_Pvalue']
            sig = '*' if row.iloc[0]['Breakpoints_Significant'] else ''
            if not FOR_SINA:
                ax.text(0.98, 0.98, f'F-test p={f_p:.4f}{sig}\nn={len(data):,}', transform=ax.transAxes,
                    va='top', ha='right', fontsize=16, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.set_title(f'CPT {cpt} (n={len(data):,})', fontsize=16)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Segmented Regression: {outcome_name} (Selected CPTs)\n(Green = wRVU Increase, Red = wRVU Decrease)', fontsize=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', format='svg')
    plt.show()
    print(f"Saved: {filename}")

CPT_GROUPS = {
    "21556": "Head & Neck",
    "30520": "Septoplasty",
    "31237": "Rhinologic",
    "42415": "Head & Neck",
    "42440": "Head & Neck",
    "60220": "Head & Neck",
    "60240": "Head & Neck",
}

def plot_specific_cpts(data_dict, results_df, reval_map, direction_map, magnitude_map, 
                       outcome_name, ylabel, filename, cpt_list):
    """
    Poster-ready segmented regression plots for selected CPTs.
    3 columns × 2 rows, large fonts, reference lines from CSV, scaled y-axes.
    """

    cpts_to_plot = [cpt for cpt in cpt_list if cpt in data_dict]
    
    if len(cpts_to_plot) == 0:
        print(f"None of the specified CPTs found in data: {cpt_list}")
        return
    
    n_cols = 3
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    axes = axes.flatten()
    
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
        
        break_years = reval_map.get(cpt, [])
        yearly_means = data.groupby('YEAR')['VALUE'].mean()
        
        # ── Plot observed data ──
        ax.plot(yearly_means.index, yearly_means.values, 'o-', 
               color='steelblue', alpha=0.8, markersize=10, linewidth=2.5, 
               label='Observed (yearly mean)', zorder=3)
        
        # ── Fit and plot segmented regression ──
        try:
            model, slopes, _ = fit_segmented(data, break_years, 'VALUE')
            years_range = np.arange(int(data['YEAR'].min()), int(data['YEAR'].max()) + 1)
            X_pred = pd.DataFrame({'YEAR': years_range})
            X_pred['const'] = 1
            for by in break_years:
                X_pred[f'TIME_SINCE_{by}'] = np.maximum(0, years_range - by)
            predictions = model.predict(X_pred)
            ax.plot(years_range, predictions, '-', color='#c0392b', 
                   linewidth=3, alpha=0.9, label='Segmented Fit', zorder=2)
        except:
            pass
        
        # ── Reference line from CSV ──
        if ref_time is not None:
            ax.axhline(y=ref_time, color='#C59E01', linestyle='--', linewidth=3, 
                      alpha=0.7, label=f'RUC Intra Time ({ref_time} min)')
        
        # ── Breakpoint lines ──
        for by in break_years:
            color = get_line_color(cpt, by, direction_map)
            ax.axvline(x=by, color=color, linestyle='--', linewidth=2, alpha=0.7, zorder=1)
        
        # ── Stats annotation ──
        row = results_df[results_df['CPT'] == cpt] if len(results_df) > 0 else None
        if row is not None and len(row) > 0:
            f_p = row.iloc[0]['F_Pvalue']
            sig = '*' if row.iloc[0]['Breakpoints_Significant'] else ''
            if not FOR_SINA:
                ax.text(0.98, 0.96, f'F-test p={f_p:.4f}{sig}\nn={len(data):,}', 
                       transform=ax.transAxes, va='top', ha='right', fontsize=14,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                edgecolor='gray', alpha=0.9))
        
        # ── Y-axis scaling: minimum 5-minute range ──
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
        
        # If reference time is outside range, extend to include it
        if ref_time is not None:
            y_min = min(y_min, ref_time - 1)
            y_max = max(y_max, ref_time + 1)
        
        y_min = max(0, y_min)
        ax.set_ylim(y_min, y_max)
        
        # ── Formatting ──
        ax.set_xlabel('Year', fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_title(f'CPT {cpt}: {group}', fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Integer x-axis
        x_min, x_max = int(yearly_means.index.min()), int(yearly_means.index.max())
        tick_step = 2
        ax.set_xticks(range(x_min, x_max + 1, tick_step))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Hide unused panels
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Segmented Regression: {outcome_name}\n'
                f'(Green = wRVU Increase, Red = wRVU Decrease, Dotted = RUC Reference Time)',
                fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', format='svg')
    plt.show()
    print(f"Saved: {filename}")

# MAIN

def main():
    print("HCUP SEGMENTED REGRESSION ANALYSIS")
    cpt_list = 'hcup_cpt_counts.csv'
    nsqip_file = 'combined_filtered_930.csv'
    ref_file = "filtered_sina2.csv"

    # Load data
    df_optime = load_optime_data('HCUP_filtered_172_cleaned.csv')
    df_volume = extract_yearly_wrvu(cpt_list, nsqip_file, output_file = 'yearly_wrvu.csv')
    reval_map, direction_map, magnitude_map = detect_revaluations_from_data(df_volume)

    # to get right range for hcup (2008, 2017)
    filtered_reval_map = {}
    for cpt, years in reval_map.items():
        valid_years = [y for y in years if 2008 <= y <= 2017]
        if valid_years:
            filtered_reval_map[cpt] = valid_years
    print(f"\nOriginal CPTs with detected revals: {len(reval_map)}")
    print(f"CPTs with breakpoints inside 2008-2017: {len(filtered_reval_map)}")
    
    print(f"Found {len(filtered_reval_map)} CPTs with revaluations (≥{MIN_RVU_CHANGE_PCT}% change):")
    for cpt, years in list(filtered_reval_map.items())[:15]:
        dir_strs = []
        for y in years:
            d, m = get_revaluation_info(cpt, y, direction_map, magnitude_map)
            arrow = '↑' if d == 'increase' else '↓' if d == 'decrease' else '?'
            dir_strs.append(f"{y}{arrow}({m:.0f}%)")
        print(f"  CPT {cpt}: {', '.join(dir_strs)}")
    
    print("OPERATIVE TIME ANALYSIS")    
    optime_results = []
    optime_data_dict = {}
    
    for cpt, break_years in filtered_reval_map.items():
        if cpt not in df_optime['CPT1'].unique():
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
        plot_results(optime_data_dict, optime_df, filtered_reval_map, direction_map,
                    "Operative Time Response", "Operative Time (minutes)", "segmented_optime_dynamic.svg")
        optime_df.to_csv('optime_segmented_results_dynamic.csv', index=False)

    target_cpts = ['21556', '30520', '38542', '42415', '42420', '42440', '60220', '60240']
    plot_specific_cpts(
        optime_data_dict, 
        optime_df, 
        filtered_reval_map, 
        direction_map,
        magnitude_map,
        "Operative Time Response", 
        "Operative Time (minutes)", 
        "segmented_optime_selected_cpts.svg",
        target_cpts
    )
    
    # FINAL SUMMARY
    print("FINAL SUMMARY")
    
    inc_count = sum(1 for cpt, years in filtered_reval_map.items() 
                    for y in years if direction_map.get(cpt, {}).get(y) == 'increase')
    dec_count = sum(1 for cpt, years in filtered_reval_map.items() 
                    for y in years if direction_map.get(cpt, {}).get(y) == 'decrease')
    
    print(f"Total revaluation events detected: {inc_count + dec_count}")
    print(f"  wRVU INCREASES (↑): {inc_count}")
    print(f"  wRVU DECREASES (↓): {dec_count}")
    print(f"\nOperative Time: {len(optime_df[optime_df['Breakpoints_Significant']==True])}/{len(optime_df)} significant")
    print("\nSaved: optime_segmented_results_dynamic.csv")
    print("Saved: segmented_optime_dynamic.svg")

if __name__ == "__main__":
    main()
