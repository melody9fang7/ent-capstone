from nsqip_segmented_lr import *
# For each CPT code:
# Standardized Volume = (Count of that CPT in that year / Total NSQIP cases that year) × 100

FOR_SINA = True
CPT_GROUPS = {
    '38542': 'Neck Dissection',
    '42415': 'Salivary Gland',
    '42420': 'Salivary Gland',
    '42440': 'Salivary Gland',
    '60220': 'Thyroid',
    '60240': 'Thyroid',
}

GROUP_COLORS = {
    'Neck Dissection': '#00a6ed',
    'Salivary Gland': '#ffa600',
    'Thyroid': '#bc4c96',
}

def load_total_cases_per_year(filepath, ent_codes):
    """
    calc total NSQIP cases per year (denominator for standardization).
    Each row in combined_filtered.csv is one case.
    """
    df = pd.read_csv(filepath, low_memory=False)
    df['YEAR'] = pd.to_numeric(df['PUFYEAR'], errors='coerce')
    df = df[(df['YEAR'] >= YEAR_START) & (df['YEAR'] <= YEAR_END)]
    
    total_per_year = df.groupby('YEAR').size().reset_index(name='total_cases')
    print(f"Total cases per year (all specialties):")
    print(total_per_year.to_string(index=False))
    return total_per_year


def build_volume_data(filepath, ent_codes, total_per_year):
    """
    for each ENT CPT code, calculate standardized volume per year.
    ALL cases (solo and non-solo) where this CPT appears as primary.
    Returns 
        dict: {cpt: DataFrame with YEAR, VALUE (standardized percentage)}
    """
    df = pd.read_csv(filepath, low_memory=False)
    df['YEAR'] = pd.to_numeric(df['PUFYEAR'], errors='coerce')
    df['CPT_NUM'] = pd.to_numeric(df['CPT'], errors='coerce')
    
    # ENT codes
    df_ent = df[df['CPT_NUM'].isin(ent_codes)].copy()
    df_ent = df_ent.dropna(subset=['YEAR'])
    df_ent = df_ent[(df_ent['YEAR'] >= YEAR_START) & (df_ent['YEAR'] <= YEAR_END)]
    
    # occurrences per CPT per year
    yearly_counts = df_ent.groupby(['CPT_NUM', 'YEAR']).size().reset_index(name='count')
    
    # merge with total cases and standardize
    yearly_counts = yearly_counts.merge(total_per_year, on='YEAR')
    yearly_counts['VALUE'] = yearly_counts['count'] / yearly_counts['total_cases'] * 100
    
    # build dict similar to optime_data_dict
    volume_data_dict = {}
    for cpt_num in yearly_counts['CPT_NUM'].unique():
        cpt_str = str(int(cpt_num))
        cpt_data = yearly_counts[yearly_counts['CPT_NUM'] == cpt_num][['YEAR', 'VALUE']].copy()
        if len(cpt_data) >= 5:
            volume_data_dict[cpt_str] = cpt_data
    
    print(f"Built volume data for {len(volume_data_dict)} CPTs")
    return volume_data_dict

def build_volume_data_combined(volume_adult, volume_peds, total_adult, total_peds, ent_codes):
    """
    Build volume data dict from combined adult + pediatric NSQIP.
    Standardized by COMBINED total cases (NSQIP + NSQIP-P).
    """
    # Combined denominator: sum total cases from both datasets per year
    total_combined = pd.concat([total_adult, total_peds]).groupby('YEAR').sum().reset_index()
    
    print(f"\nCombined total cases per year (NSQIP + NSQIP-P):")
    print(total_combined.to_string(index=False))
    
    # Raw counts per CPT per year from adult
    adult_counts = volume_adult.groupby(['CPT_NUM', 'YEAR']).size().reset_index(name='count_adult')
    
    # Raw counts per CPT per year from peds
    peds_counts = volume_peds.groupby(['CPT_NUM', 'YEAR']).size().reset_index(name='count_peds')
    
    # Merge and sum counts from both datasets
    yearly_counts = adult_counts.merge(peds_counts, on=['CPT_NUM', 'YEAR'], how='outer')
    yearly_counts['count_adult'] = yearly_counts['count_adult'].fillna(0).astype(int)
    yearly_counts['count_peds'] = yearly_counts['count_peds'].fillna(0).astype(int)
    yearly_counts['count'] = yearly_counts['count_adult'] + yearly_counts['count_peds']
    
    # Standardize by COMBINED total
    yearly_counts = yearly_counts.merge(total_combined, on='YEAR')
    yearly_counts['VALUE'] = yearly_counts['count'] / yearly_counts['total_cases'] * 100
    
    # Build dict
    volume_data_dict = {}
    for cpt_num in yearly_counts['CPT_NUM'].unique():
        cpt_str = str(int(cpt_num))
        cpt_data = yearly_counts[yearly_counts['CPT_NUM'] == cpt_num][['YEAR', 'VALUE']].copy()
        if len(cpt_data) >= 5:
            volume_data_dict[cpt_str] = cpt_data
    
    print(f"Built combined volume data for {len(volume_data_dict)} CPTs")
    print(f"(Standardized by NSQIP + NSQIP-P total cases per year)")
    return volume_data_dict

def plot_results(data_dict, results_df, reval_map, direction_map, outcome_name, ylabel, filename):
    """ segmented regression volume plots. """
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
                ax.axvline(x=by, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
        except:
            ax.plot(yearly_means.index, yearly_means.values, 'o-', color='steelblue', alpha=0.7)
        
        #  Y-axis scaling 
        y_min_data = yearly_means.values.min()
        y_max_data = yearly_means.values.max()
        y_range = y_max_data - y_min_data
        
        if y_range < 0.01:
            y_center = (y_max_data + y_min_data) / 2
            y_min = max(0, y_center - 0.005)
            y_max = y_center + 0.005
        else:
            padding = y_range * 0.15
            y_min = max(0, y_min_data - padding)
            y_max = y_max_data + padding
        
        ax.set_ylim(y_min, y_max)
        
        #  Stats 
        row = results_df[results_df['CPT'] == cpt]
        if len(row) > 0:
            f_p = row.iloc[0]['F_Pvalue']
            sig = '*' if row.iloc[0]['Breakpoints_Significant'] else ''
            if not FOR_SINA:
                ax.text(0.98, 0.96, f'F-test p={f_p:.4f}{sig}\nn={len(data):,}', 
                    transform=ax.transAxes, va='top', ha='right', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        ax.set_xlabel('Year', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(f'CPT {cpt} (n={len(data):,})', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        
        # Integer x-axis
        x_min, x_max = int(yearly_means.index.min()), int(yearly_means.index.max())
        tick_step = max(1, (x_max - x_min) // 4)
        ax.set_xticks(range(x_min, x_max + 1, tick_step))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Segmented Regression: {outcome_name}\n'
                f'(Green = wRVU Increase, Red = wRVU Decrease)',
                fontsize=18, fontweight='bold')
    plt.savefig(filename, dpi=200, facecolor='white', format='svg')
    plt.show()
    print(f"Saved: {filename}")


def run_volume_analysis(reval_map, direction_map, magnitude_map, volume_data_dict):
    volume_results = []
    
    for cpt, break_years in reval_map.items():
        if cpt not in volume_data_dict:
            continue
        
        data = volume_data_dict[cpt]
        if len(data) < 5:
            continue        
        valid_breaks = [by for by in break_years 
                       if data['YEAR'].min() <= by <= data['YEAR'].max()]
        
        if not valid_breaks:
            continue
        
        result = evaluate_breakpoints(data, cpt, valid_breaks, 'VALUE', 'Volume')
        if result:
            volume_results.append(result)
    
    volume_df = pd.DataFrame(volume_results)
    
    if len(volume_df) > 0:
        print_results_table(volume_df, "Procedural Volume", direction_map, magnitude_map)
        print_detailed_results(volume_df, direction_map, magnitude_map)
        plot_results(volume_data_dict, volume_df, reval_map, direction_map,
                    "Procedural Volume Response", 
                    "Percent of Total NSQIP Cases", 
                    "segmented_volume_dynamic.svg")
        volume_df.to_csv('volume_segmented_results_dynamic.csv', index=False)
    
    return volume_df


def plot_specific_cpts(data_dict, results_df, reval_map, direction_map, magnitude_map, 
                       outcome_name, ylabel, filename, cpt_list):
    """
    Volume segmented regression plots for selected CPTs.
    3 columns x 2 rows, matches operative time style.
    """
    cpts_to_plot = [cpt for cpt in cpt_list if cpt in data_dict]
    
    if len(cpts_to_plot) == 0:
        print(f"None of the specified CPTs found in data: {cpt_list}")
        return
    
    n_cols = 3
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    axes = axes.flatten()
    
    # For CSV export
    export_rows = []
    
    for idx, cpt in enumerate(cpts_to_plot):
        ax = axes[idx]
        data = data_dict.get(cpt)
        group = CPT_GROUPS.get(cpt, '')
        
        if data is None or len(data) < 6:
            ax.text(0.5, 0.5, f'CPT {cpt}: {group}\nInsufficient data', 
                   ha='center', va='center', fontsize=18)
            ax.set_title(f'CPT {cpt}: {group}', fontsize=22, fontweight='bold')
            continue
        
        break_years = reval_map.get(cpt, [])
        yearly_means = data.groupby('YEAR')['VALUE'].mean()
        
        # Build export rows
        for year, val in yearly_means.items():
            export_rows.append({
                'CPT': cpt,
                'Group': group,
                'Year': int(year),
                'Yearly_Mean': round(val, 6),
                'Break_Years': ', '.join(str(b) for b in break_years) if break_years else '',
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
        
        if y_range < 0.01:
            y_center = (y_max_data + y_min_data) / 2
            y_min = max(0, y_center - 0.005)
            y_max = y_center + 0.005
        else:
            padding = y_range * 0.15
            y_min = max(0, y_min_data - padding)
            y_max = y_max_data + padding
        
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
        Line2D([0], [0], color='green', linestyle='--', linewidth=3, label='wRVU Increase'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=3, label='wRVU Decrease'),
    ]
    
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, 
              fontsize=16, frameon=True, bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle(f'Segmented Regression: {outcome_name}',
                fontsize=24, fontweight='bold', y=1.01)
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.10, 
                        hspace=0.35, wspace=0.25)
    
    plt.savefig(filename, dpi=300, facecolor='white', format='svg', bbox_inches="tight", pad_inches=0.3)
    plt.show()
    
    # Export to CSV
    export_df = pd.DataFrame(export_rows)
    csv_filename = filename.replace('.svg', '_data.csv')
    export_df.to_csv(csv_filename, index=False)
    
    print(f"Saved: {filename}")
    print(f"Saved: {csv_filename}")

def export_reval_maps(reval_map, direction_map, magnitude_map, filename='reval_breakpoints.csv'):
    # really only needed to run this once so that we can project revaluation years onto MNPB data
    # without running the actual whole NSQIP files over and over again
    """
    evaluation breakpoints, directions, and magnitudes to CSV.
    one row per CPT per breakpoint year.
    """
    rows = []
    for cpt, years in reval_map.items():
        for year in years:
            direction = direction_map.get(cpt, {}).get(year, 'unknown')
            magnitude = magnitude_map.get(cpt, {}).get(year, 0)
            rows.append({
                'CPT': cpt,
                'Year': year,
                'Direction': direction,
                'Magnitude_%': round(magnitude, 1),
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['CPT', 'Year'])
    df.to_csv(filename, index=False)
    
    print(f"\nExported {len(df)} revaluation events to {filename}")
    
    print("\nCOPY-PASTE DICTIONARIES")
    print(f"REVAL_BREAKPOINTS = {reval_map}")
    print(f"\nREVAL_DIRECTIONS = {direction_map}")
    
    return df

def plot_single_cpt_volume(volume_data_dict, cpt, results_df, reval_map, direction_map,
                           ylabel='% of Total NSQIP Cases', filename=None):
    """
    Plot a single CPT from NSQIP volume data with segmented regression and breakpoints.
    Large fonts, legend at bottom.
    """
    if filename is None:
        filename = f'nsqip_single_{cpt}.svg'
    
    if cpt not in volume_data_dict:
        print(f"CPT {cpt} not in volume data")
        return
    
    data = volume_data_dict[cpt].sort_values('YEAR')
    break_years = reval_map.get(cpt, [])
    group = CPT_GROUPS.get(cpt, '')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot observed data 
    yearly_means = data.groupby('YEAR')['VALUE'].mean()
    ax.plot(yearly_means.index, yearly_means.values, 'o-', 
           color='steelblue', alpha=0.9, markersize=12, linewidth=3, 
           label='NSQIP Volume', zorder=3)
    
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
    
    #  Stats annotation 
    if results_df is not None:
        row = results_df[results_df['CPT'] == cpt]
        if len(row) > 0:
            f_p = row.iloc[0]['F_Pvalue']
            sig = '*' if row.iloc[0]['Breakpoints_Significant'] else ''
            r2 = row.iloc[0]['R2_Segmented']
            ax.text(0.96, 0.96, f'F-test p={f_p:.4f}{sig}\nR²={r2:.4f}', 
                   transform=ax.transAxes, va='top', ha='right', fontsize=24,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='gray', alpha=0.9))
    
    #  Y-axis scaling 
    y_min, y_max = yearly_means.min(), yearly_means.max()
    y_range = y_max - y_min
    padding = max(y_range * 0.15, 0.001)
    ax.set_ylim(max(0, y_min - padding), y_max + padding)
    
    #  Formatting 
    ax.set_xlabel('Year', fontsize=28, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=28, fontweight='bold')
    ax.set_title(f'CPT {cpt}: {group} — Volume', 
                fontsize=35, fontweight='bold')
    ax.tick_params(labelsize=22)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Integer x-axis
    x_min, x_max = int(yearly_means.index.min()), int(yearly_means.index.max())
    tick_step = max(1, (x_max - x_min) // 8)
    ax.set_xticks(range(x_min, x_max + 1, tick_step))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    plt.savefig(filename, dpi=250, facecolor='white', format='svg')
    plt.show()
    print(f"Saved: {filename}")

def volume_main_combined():
    print("SEGMENTED REGRESSION ANALYSIS — PROCEDURAL VOLUME")
    
    ent_codes = load_ent_codes('nsqip_cleaning/ENT_CPT_CODES.csv')
    
    # ═══════════════════════════════════════════════════════════
    # CHOOSE ONE:
    # ═══════════════════════════════════════════════════════════
    
    # ── OPTION A: NSQIP Adult only ──
    df_volume = load_data_for_reval('nsqip_cleaning/combined_filtered.csv', ent_codes)
    total_per_year = load_total_cases_per_year('nsqip_cleaning/combined_filtered.csv', ent_codes)
    volume_data_dict = build_volume_data('nsqip_cleaning/combined_filtered.csv', ent_codes, total_per_year)
    
    # ── OPTION B: NSQIP-P Pediatric only ──
    # df_volume = load_data_for_reval('nsqip-pediatrics/ALL_NSQIP-P.csv', ent_codes)
    # total_per_year = load_total_cases_per_year('nsqip-pediatrics/ALL_NSQIP-P.csv', ent_codes)
    # volume_data_dict = build_volume_data('nsqip-pediatrics/ALL_NSQIP-P.csv', ent_codes, total_per_year)
    
    # ── OPTION C: Combined Adult + Pediatric ──
    #print("Loading NSQIP Adult volume data: ")
    #volume_adult = load_data_for_reval('nsqip_cleaning/combined_filtered.csv', ent_codes)
    #total_adult = load_total_cases_per_year('nsqip_cleaning/combined_filtered.csv', ent_codes)
    #vol_dict_adult = build_volume_data('nsqip_cleaning/combined_filtered.csv', ent_codes, total_adult)
    
    #print("\nLoading NSQIP-P Pediatric volume data: ")
    #volume_peds = load_data_for_reval('nsqip-pediatrics/ALL_NSQIP-P.csv', ent_codes)
    #total_peds = load_total_cases_per_year('nsqip-pediatrics/ALL_NSQIP-P.csv', ent_codes)
    #vol_dict_peds = build_volume_data('nsqip-pediatrics/ALL_NSQIP-P.csv', ent_codes, total_peds)
    
    # Combine revaluation data
    #df_volume = pd.concat([volume_adult, volume_peds], ignore_index=True)
    #print(f"\nCombined volume data for revaluation detection: {len(df_volume):,} rows")
    
    # Combine total cases per year (sum both datasets)
    #total_combined = pd.concat([total_adult, total_peds]).groupby('YEAR').sum().reset_index()
    #print(f"Combined total cases per year:")
    #print(total_combined.to_string(index=False))
    
    # Combine volume data dicts (add the rates together? no — recalculate from combined)
    # Since each CPT appears in both datasets with different denominators,
    # we need to rebuild from the combined raw data.
    # Simpler approach: combine the raw counts and recalculate
    
    #Rebuild volume dict from combined data
    #volume_data_dict = build_volume_data_combined(
    #    volume_adult, volume_peds, total_adult, total_peds, ent_codes
    #)

    # ── ──────── Everything after this should be the same ──────── ──

    reval_map, direction_map, magnitude_map = detect_revaluations_from_data(df_volume)

    #export_reval_maps(reval_map, direction_map, magnitude_map, 'reval_breakpoints_VOLUME.csv')

    
    print(f"\nFound {len(reval_map)} CPTs with revaluations (≥{MIN_RVU_CHANGE_PCT}% change)")
    
    print("\nPROCEDURAL VOLUME ANALYSIS")
    volume_df = run_volume_analysis(reval_map, direction_map, magnitude_map, volume_data_dict)

    target_cpts = ['38542', '42415', '42420', '42440', '60220', '60240']
    plot_specific_cpts(
        volume_data_dict, 
        volume_df, 
        reval_map, 
        direction_map,
        magnitude_map,
        "Procedural Volume Response", 
        "Percent of Total NSQIP Cases", 
        "segmented_volume_selected_combined.svg",
        target_cpts
    )
    
    print("\nSaved: volume_segmented_results_dynamic.csv")
    print("Saved: segmented_volume_dynamic.svg")
    print("Saved: segmented_volume_selected_combined.svg")

    plot_single_cpt_volume(
        volume_data_dict, '42420', volume_df, reval_map, direction_map,
        ylabel='% of Total NSQIP Cases',
        filename='nsqip_single_42420_salivary.svg'
    )


if __name__ == "__main__":
    volume_main_combined()
