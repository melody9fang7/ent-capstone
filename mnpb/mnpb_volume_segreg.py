import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


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

TARGET_CPTS = ['38542', '42415', '42420', '42440', '60220', '60240']
FOR_SINA = True


# LOAD

# Replace the hardcoded dictionaries with this:

def load_reval_breakpoints(filepath='reval_breakpoints_VOLUME.csv'):
    """
    Load revaluation breakpoints from CSV.
    Expected columns: CPT, Year, Direction, Magnitude_%
    Returns: reval_map, direction_map, magnitude_map
    """
    df = pd.read_csv(filepath)
    df['CPT'] = df['CPT'].astype(str).str.strip()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Magnitude_%'] = pd.to_numeric(df['Magnitude_%'], errors='coerce')
    
    reval_map = {}
    direction_map = {}
    magnitude_map = {}
    
    for _, row in df.iterrows():
        cpt = row['CPT']
        year = int(row['Year'])
        direction = row['Direction']
        magnitude = row['Magnitude_%']
        
        if cpt not in reval_map:
            reval_map[cpt] = []
            direction_map[cpt] = {}
            magnitude_map[cpt] = {}
        
        reval_map[cpt].append(year)
        direction_map[cpt][year] = direction
        magnitude_map[cpt][year] = magnitude
    
    print(f"Loaded revaluation breakpoints for {len(reval_map)} CPTs from {filepath}")
    return reval_map, direction_map, magnitude_map

def load_ent_codes(filepath):
    df = pd.read_csv(filepath)
    codes = set()
    for val in df['CPT Code']:
        try:
            if pd.notna(val):
                codes.add(str(int(float(str(val).strip()))))
        except:
            continue
    print(f"Loaded {len(codes)} ENT CPT codes")
    return codes


def load_mnpb(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df):,} MNPB rows")
    
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    df['HCPCS'] = df['HCPCS'].astype(str).str.strip()
    df['MODIFIER'] = df['MODIFIER'].astype(str).str.strip()
    
    for col in ['ALLOWED SERVICES', 'ALLOWED CHARGES', 'PAYMENT']:
        df[col] = df[col].astype(str).str.replace('$', '', regex=False)
        df[col] = df[col].str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[(df['YEAR'] >= YEAR_START) & (df['YEAR'] <= YEAR_END)]
    return df


# SEGMENTED REGRESSION

def fit_segmented(data, break_years, outcome_col):
    """Fit segmented regression with slope changes at breakpoints."""
    data = data.sort_values('YEAR').copy()
    X = data[['YEAR']].copy()
    X['const'] = 1
    
    for by in break_years:
        col_name = f'TIME_SINCE_{by}'
        data[col_name] = np.maximum(0, data['YEAR'] - by)
        X[col_name] = data[col_name]
    
    model = sm.OLS(data[outcome_col], X).fit()
    
    slopes = [model.params['YEAR']]
    slope_changes = {}
    for by in break_years:
        sc = model.params.get(f'TIME_SINCE_{by}', 0)
        slope_changes[by] = sc
        slopes.append(slopes[-1] + sc)
    
    return model, slopes, slope_changes


def evaluate_breakpoints(data, cpt, break_years, outcome_col, outcome_name):
    """Evaluate if segmented model fits better than simple linear."""
    if not break_years or len(data) < 6:
        return None
    
    data = data.sort_values('YEAR')
    
    # Simple linear model
    X_simple = sm.add_constant(data['YEAR'])
    simple_model = sm.OLS(data[outcome_col], X_simple).fit()
    
    # Segmented model
    try:
        seg_model, slopes, slope_changes = fit_segmented(data, break_years, outcome_col)
    except:
        return None
    
    # F-test
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
        'CPT': cpt,
        'Break_Years': break_years,
        'Outcome': outcome_name,
        'n': len(data),
        'R2_Simple': simple_model.rsquared,
        'R2_Segmented': seg_model.rsquared,
        'RSS_Reduction_Pct': rss_reduction_pct,
        'F_Pvalue': f_pvalue,
        'Breakpoints_Significant': f_pvalue < 0.05,
        'Pre_Slope': slopes[0],
        'Slope_Changes': slope_changes,
        'Slope_Pvalues': slope_pvalues,
    }


# ============================================================================
# CALCULATE MNPB METRICS
# ============================================================================

def calculate_mnpb_metrics(df_all, ent_codes):
    """
    Calculate volume, average allowed charge, and average payment per CPT per year.
    All use TOTAL modifier.
    """
    # Denominator: total Part B services
    total_all = df_all[(df_all['MODIFIER'] == 'TOTAL') & (df_all['ALLOWED SERVICES'].notna())]
    total_per_year = total_all.groupby('YEAR')['ALLOWED SERVICES'].sum().reset_index(name='total_services')
    
    # ENT data
    df_ent = df_all[
        (df_all['HCPCS'].isin(ent_codes)) & 
        (df_all['MODIFIER'] == 'TOTAL') &
        (df_all['ALLOWED SERVICES'].notna())
    ]
    
    yearly = df_ent.groupby(['HCPCS', 'YEAR']).agg(
        services=('ALLOWED SERVICES', 'sum'),
        total_payment=('PAYMENT', 'sum'),
        total_charges=('ALLOWED CHARGES', 'sum')
    ).reset_index()
    
    yearly = yearly.merge(total_per_year, on='YEAR')
    
    # Metrics
    yearly['volume_pct'] = yearly['services'] / yearly['total_services'] * 100
    yearly['avg_payment'] = yearly['total_payment'] / yearly['services']
    yearly['avg_charge'] = yearly['total_charges'] / yearly['services']
    
    print(f"ENT CPTs with MNPB data: {yearly['HCPCS'].nunique()}")
    return yearly


# ============================================================================
# RUN SEGMENTED REGRESSION ON VOLUME
# ============================================================================

def run_volume_segmented(yearly, cpt_list, reval_map, direction_map, magnitude_map):
    """Run segmented regression on MNPB volume for selected CPTs."""
    
    results_list = []
    
    for cpt in cpt_list:
        if cpt not in yearly['HCPCS'].values:
            continue
        
        cpt_data = yearly[yearly['HCPCS'] == cpt].sort_values('YEAR')
        break_years = reval_map.get(cpt, [])
        
        # Only keep breakpoints within data range
        valid_breaks = [by for by in break_years 
                       if cpt_data['YEAR'].min() <= by <= cpt_data['YEAR'].max()]
        
        if len(valid_breaks) == 0 or len(cpt_data) < 6:
            continue
        
        result = evaluate_breakpoints(cpt_data, cpt, valid_breaks, 'volume_pct', 'MNPB Volume')
        if result:
            results_list.append(result)
    
    if len(results_list) == 0:
        print("No results to display")
        return None
    
    results_df = pd.DataFrame(results_list)
    
    # Print results
    print(f"\n{'='*80}")
    print("MNPB VOLUME — SEGMENTED REGRESSION RESULTS")
    print(f"{'='*80}")
    print(f"{'CPT':<8} {'Break Years':<25} {'n':<5} {'F-test p':<10} {'Sig':<5} {'RSS Red%':<10} {'R² Simple':<10} {'R² Seg':<10}")
    print("-"*80)
    
    for _, row in results_df.iterrows():
        # Format break years with direction
        by_strs = []
        for by in row['Break_Years']:
            direction = direction_map.get(row['CPT'], {}).get(by, '?')
            mag = magnitude_map.get(row['CPT'], {}).get(by, 0)
            arrow = '↑' if direction == 'increase' else '↓'
            by_strs.append(f"{by}{arrow}({mag:.0f}%)")
        
        sig = '✓' if row['Breakpoints_Significant'] else '✗'
        print(f"{row['CPT']:<8} {', '.join(by_strs):<25} {row['n']:<5} {row['F_Pvalue']:.4f}     {sig:<5} {row['RSS_Reduction_Pct']:.1f}%       {row['R2_Simple']:.4f}     {row['R2_Segmented']:.4f}")
    
    sig_count = results_df['Breakpoints_Significant'].sum()
    print(f"\nSignificant: {sig_count}/{len(results_df)} ({sig_count/len(results_df)*100:.1f}%)")
    
    return results_df


# PLOT

def plot_mnpb_segmented(yearly, results_df, cpt_list, reval_map, direction_map, 
                         outcome_col='volume_pct', ylabel='% of All Part B Services',
                         filename='mnpb_segmented_volume.svg', max_cols=3):
    """Plot MNPB data with segmented regression fit and breakpoints."""
    
    cpts_to_plot = [c for c in cpt_list if c in yearly['HCPCS'].values]
    
    if len(cpts_to_plot) == 0:
        print("No CPTs to plot")
        return
    
    n_plots = len(cpts_to_plot)
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # For CSV export
    export_rows = []
    
    for idx, cpt in enumerate(cpts_to_plot):
        ax = axes[idx]
        cpt_data = yearly[yearly['HCPCS'] == cpt].sort_values('YEAR')
        group = CPT_GROUPS.get(cpt, '')
        break_years = reval_map.get(cpt, [])
        
        if len(cpt_data) < 3:
            continue
        
        # Export the raw data being plotted
        for _, row in cpt_data.iterrows():
            export_rows.append({
                'CPT': cpt,
                'Group': group,
                'Year': int(row['YEAR']),
                'Volume_pct': round(row[outcome_col], 6),
                'Break_Years': ', '.join(str(b) for b in break_years) if break_years else '',
            })
        
        # Plot observed
        ax.plot(cpt_data['YEAR'], cpt_data[outcome_col], 'o-', 
               color='steelblue', alpha=0.8, markersize=8, linewidth=2, 
               label='Observed', zorder=3)
        
        # Fit and plot segmented regression
        try:
            model, slopes, _ = fit_segmented(cpt_data, break_years, outcome_col)
            years_range = np.arange(int(cpt_data['YEAR'].min()), int(cpt_data['YEAR'].max()) + 1)
            X_pred = pd.DataFrame({'YEAR': years_range})
            X_pred['const'] = 1
            for by in break_years:
                X_pred[f'TIME_SINCE_{by}'] = np.maximum(0, years_range - by)
            predictions = model.predict(X_pred)
            ax.plot(years_range, predictions, '-', color='#c0392b', 
                   linewidth=2.5, alpha=0.9, label='Segmented Fit', zorder=2)
        except:
            pass
        
        # Breakpoint lines
        for by in break_years:
            direction = direction_map.get(cpt, {}).get(by, None)
            color = 'green' if direction == 'increase' else 'red' if direction == 'decrease' else 'gray'
            ax.axvline(x=by, color=color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
        
        # Stats
        if results_df is not None and not FOR_SINA:
            row = results_df[results_df['CPT'] == cpt]
            if len(row) > 0:
                f_p = row.iloc[0]['F_Pvalue']
                sig = '*' if row.iloc[0]['Breakpoints_Significant'] else ''
                ax.text(0.98, 0.96, f'p={f_p:.4f}{sig}', 
                       transform=ax.transAxes, va='top', ha='right', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        # Formatting
        y_data = cpt_data[outcome_col]
        y_min, y_max = y_data.min(), y_data.max()
        y_range = y_max - y_min
        padding = max(y_range * 0.15, 0.0005)
        ax.set_ylim(max(0, y_min - padding), y_max + padding)
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'CPT {cpt}', fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        x_min, x_max = int(cpt_data['YEAR'].min()), int(cpt_data['YEAR'].max())
        tick_step = max(1, (x_max - x_min) // 4)
        ax.set_xticks(range(x_min, x_max + 1, tick_step))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Hide unused panels
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='steelblue', linewidth=2, label='MNPB Volume'),
        Line2D([0], [0], color='#c0392b', linewidth=2, label='Segmented Fit'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=1.5, label='wRVU Increase'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='wRVU Decrease'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=12, 
              frameon=True, bbox_to_anchor=(0.5, -0.01))
    
    plt.suptitle('MNPB Volume — Segmented Regression with wRVU Breakpoints\n'
                '(Medicare Part B, TOTAL Modifier)',
                fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Export CSV
    export_df = pd.DataFrame(export_rows)
    if '.svg' in filename:
        csv_filename = filename.replace('.svg', '_data.csv')
    elif '.png' in filename:
        csv_filename = filename.replace('.png', '_data.csv')
    else:
        csv_filename = filename + '_data.csv'
    export_df.to_csv(csv_filename, index=False)
    
    print(f"Saved: {filename}")
    print(f"Saved: {csv_filename}")


# MAIN

PROCEDURE_GROUPS = {
    'Glossectomies & Laryngectomies': ['31360', '31365', '41120', '41130', '41135', '41155'],
    'Other Oral Cavity Resections': ['21044', '40810', '40816', '42120', '42842'],
    'Neck Dissections': ['38542', '38700', '38720', '38724'],
    'Salivary Gland Surgeries': ['42415', '42420', '42440'],
    'Thyroid Surgeries': ['60220', '60240', '60252', '60254', '60260', '60270', '60271'],
    'Miscellaneous Codes': ['15731', '21556', '31591', '42145']
}

def main():
    print("MNPB SEGMENTED REGRESSION — Volume Response to wRVU Revaluation")
    
    reval_map, direction_map, magnitude_map = load_reval_breakpoints('reval_breakpoints_VOLUME.csv')
    ALL_CPTS = sorted(reval_map.keys())
    print(f"Retrieved {len(ALL_CPTS)} CPTs from reval_breakpoints_VOLUME.csv")
    
    ent_codes = load_ent_codes('mnpb/ENT_CPT_CODES.csv')
    df_all = load_mnpb('mnpb/MNPB_MASTER_FINAL.csv')
    
    yearly = calculate_mnpb_metrics(df_all, ent_codes)
    
    results_df = run_volume_segmented(
        yearly, ALL_CPTS, reval_map, direction_map, magnitude_map
    )
    
    if results_df is not None:
        results_df.to_csv('mnpb_volume_segmented_results_all.csv', index=False)

        # Volume Changes In Response to wRVU changes in MPNB for selected codes from both NSQIP and HCUP only
        SELECTED_NSQIP_u_HCUP = ['42415', '42440', '60220', '60240']
        plot_mnpb_segmented(
                yearly, results_df, SELECTED_NSQIP_u_HCUP, reval_map, direction_map,
                outcome_col='volume_pct', ylabel='% of All Part B Services',
               filename=f'mnpb_segmented_volume_selected_codes.svg',
                max_cols=2
        )
        plot_mnpb_segmented(
                yearly, results_df, SELECTED_NSQIP_u_HCUP, reval_map, direction_map,
                outcome_col='volume_pct', ylabel='% of All Part B Services',
               filename=f'mnpb_segmented_volume_selected_codes.png',
                max_cols=2
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
