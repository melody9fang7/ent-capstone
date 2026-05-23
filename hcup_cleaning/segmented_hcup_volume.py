
from segmented_hcup import *

# For each CPT code:
# Standardized Volume = (Count of that CPT in that year / Total HCUP cases that year) × 100

FOR_SINA = True

def plot_results(data_dict, results_df, reval_map, direction_map, outcome_name, ylabel, filename):
    """Create segmented regression volume plots with large fonts."""
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
        
        # ── Y-axis scaling ──
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
        
        # ── Stats ──
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
        tick_step = 2
        ax.set_xticks(range(x_min, x_max + 1, tick_step))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Segmented Regression: {outcome_name}\n'
                f'(Green = wRVU Increase, Red = wRVU Decrease)',
                fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', format='svg')
    plt.show()
    print(f"Saved: {filename}")

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
    Volume segmented regression plots for selected CPTs.
    3 columns × 2 rows, large fonts, matches operative time style.
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
               label='Observed Mean Volume', zorder=3)
        
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
                   linewidth=3, alpha=0.9, label='Regression', zorder=2)
        except:
            pass
        
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
        
        # ── Y-axis scaling ──
        y_min_data = yearly_means.values.min()
        y_max_data = yearly_means.values.max()
        y_range = y_max_data - y_min_data
        
        if y_range < 0.01:  # Very small range
            y_center = (y_max_data + y_min_data) / 2
            y_min = max(0, y_center - 0.005)
            y_max = y_center + 0.005
        else:
            padding = y_range * 0.15
            y_min = max(0, y_min_data - padding)
            y_max = y_max_data + padding
        
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
                f'(Green = wRVU Increase, Red = wRVU Decrease)',
                fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', format='svg')
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

    target_cpts = ['21556', '30520', '42415', '42440', '60220', '60240']
    plot_specific_cpts(
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
