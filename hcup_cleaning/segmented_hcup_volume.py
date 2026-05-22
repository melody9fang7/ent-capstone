
from segmented_hcup import *

# For each CPT code:
# Standardized Volume = (Count of that CPT in that year / Total HCUP cases that year) × 100
def load_volume_data(filepath):
    df = pd.read_csv(filepath, low_memory = False)
    print(f"Loaded volume time data: {len(df):,} rows")

    df["AYEAR"] = pd.to_numeric(df["AYEAR"], errors = "coerce")
    df["PRIMARY_COUNT"] = pd.to_numeric(df["PRIMARY_COUNT"], errors="coerce")
    df["NORMALIZED_VOLUME"] = pd.to_numeric(df["NORMALIZED_VOLUME"], errors="coerce")
    df["CPT"] = standardize_cpt(df["CPT"])
    df = df.dropna(subset = ["NORMALIZED_VOLUME", "AYEAR"])

    df = df[(df["AYEAR"] >= YEAR_START) & (df["AYEAR"] <= YEAR_END)]

    return df

def get_volume_cpts(volume_file, output_file, min_cases=100):
    """
    Gets CPT codes from HCUP volume table with at least min_cases primary cases.
    """
    df_volume = pd.read_csv(volume_file)

    df_volume["CPT"] = standardize_cpt(df_volume["CPT"])
    df_volume["PRIMARY_COUNT"] = pd.to_numeric(df_volume["PRIMARY_COUNT"], errors="coerce")

    counts = df_volume.groupby("CPT")["PRIMARY_COUNT"].sum()
    keep_cpts = counts[counts >= min_cases].index.tolist()

    keep_df = pd.DataFrame({"CPT1": keep_cpts})
    keep_df.to_csv(output_file, index=False)

    print(f"CPTs with >={min_cases} primary cases: {len(keep_cpts)}")
    print(f"Saved CPT list to: {output_file}")

    return output_file

def get_volume_data(df, cpt):
    data = df[df["CPT"] == cpt].copy()
    if len(data) < 5:
        return None
    return data[["AYEAR", "NORMALIZED_VOLUME"]].rename(columns={"AYEAR": "YEAR", "NORMALIZED_VOLUME": "VALUE"})

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
                    "Percent of Total HCUP Cases", 
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
    volume_file = "hcup_volume_table.csv"
    nsqip_file = "combined_filtered_930.csv"

    cpt_file = get_volume_cpts(volume_file, output_file = "volume_cpts.csv", min_cases = 100)  

    df_rvu = extract_yearly_wrvu(cpt_file, nsqip_file, output_file = "yearly_wrvu_volume.csv")
    reval_map, direction_map, magnitude_map = detect_revaluations_from_data(df_rvu)

    filtered_reval_map = {}
    for cpt, years in reval_map.items():
        valid_years = [y for y in years if 2008 <= y <= 2017]
        if valid_years:
            filtered_reval_map[cpt] = valid_years
    print(f"\nOriginal CPTs with detected revals: {len(reval_map)}")
    print(f"CPTs with breakpoints inside 2008-2017: {len(filtered_reval_map)}")

    df_volume = load_volume_data(volume_file)
    
    print("PROCEDURAL VOLUME ANALYSIS")

    volume_data_dict = {}
    for cpt in df_volume["CPT"].unique():
        data = get_volume_data(df_volume, cpt)
        if data is not None:
            volume_data_dict[cpt] = data

    volume_df = run_volume_analysis(filtered_reval_map, direction_map, magnitude_map, volume_data_dict)

    target_cpts = ['21556', '30520', '38542', '42415', '42420', '42440', '60220', '60240']
    plot_specific_cpts_volume(
        volume_data_dict, 
        volume_df, 
        filtered_reval_map, 
        direction_map,
        magnitude_map,
        "Procedural Volume Response", 
        "Percent of Total HCUP Cases", 
        "segmented_volume_selected_cpts.svg",
        target_cpts
    )

if __name__ == "__main__":
    volume_main()
