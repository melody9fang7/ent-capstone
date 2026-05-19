from nsqip_segmented_lr import *
# For each CPT code:
# Standardized Volume = (Count of that CPT in that year / Total NSQIP cases that year) × 100

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

def plot_specific_cpts_volume(data_dict, results_df, reval_map, direction_map, magnitude_map, 
                               outcome_name, ylabel, filename, cpt_list):
    cpts_to_plot = [cpt for cpt in cpt_list if cpt in data_dict]
    
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
        
        # observed data
        ax.plot(yearly_means.index, yearly_means.values, 'o-', color='steelblue', 
               alpha=0.7, markersize=4, linewidth=1.5, label='Observed')
        
        # fit and plot segmented regression
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
        
        # breakpoint lines and annotations
        y_min, y_max = ax.get_ylim()
        for i, by in enumerate(break_years):
            color = get_line_color(cpt, by, direction_map)
            ax.axvline(x=by, color=color, linestyle='--', linewidth=1.5, alpha=0.7)        
        row = results_df[results_df['CPT'] == cpt] if len(results_df) > 0 else None
        if row is not None and len(row) > 0:
            f_p = row.iloc[0]['F_Pvalue']
            sig = '*' if row.iloc[0]['Breakpoints_Significant'] else ''
            ax.text(0.98, 0.98, f'F-test p={f_p:.4f}{sig}', transform=ax.transAxes,
                    va='top', ha='right', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.set_title(f'CPT {cpt}')
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Segmented Regression: {outcome_name} (Selected CPTs)\n(Green = wRVU Increase, Red = wRVU Decrease)', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', format='svg')
    plt.show()
    print(f"Saved: {filename}")

def volume_main():
    print("SEGMENTED REGRESSION ANALYSIS")
    
    ent_codes = load_ent_codes('nsqip_cleaning/ENT_CPT_CODES.csv')
    df_volume = load_data_for_reval('nsqip_cleaning/combined_filtered.csv', ent_codes)
    
    reval_map, direction_map, magnitude_map = detect_revaluations_from_data(df_volume)
    
    print("PROCEDURAL VOLUME ANALYSIS")
    
    total_per_year = load_total_cases_per_year('nsqip_cleaning/combined_filtered.csv', ent_codes)
    volume_data_dict = build_volume_data('nsqip_cleaning/combined_filtered.csv', ent_codes, total_per_year)
    volume_df = run_volume_analysis(reval_map, direction_map, magnitude_map, volume_data_dict)

    target_cpts = ['38542', '42415', '42420', '42440', '60220', '60240']
    plot_specific_cpts_volume(
        volume_data_dict, 
        volume_df, 
        reval_map, 
        direction_map,
        magnitude_map,
        "Procedural Volume Response", 
        "Percent of Total NSQIP Cases", 
        "segmented_volume_selected_cpts.svg",
        target_cpts
    )

if __name__ == "__main__":
    volume_main()