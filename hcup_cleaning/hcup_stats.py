from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from filtering import standardize_cpt
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

group_map = {
    # Septoplasty and turbinectomy
    '21315': 'Septoplasty/Turbinectomy/Closed Nasal Bone Reduction',
    '21320': 'Septoplasty/Turbinectomy/Closed Nasal Bone Reduction',
    '30520': 'Septoplasty/Turbinectomy/Closed Nasal Bone Reduction',
    '30140': 'Septoplasty/Turbinectomy/Closed Nasal Bone Reduction',
    '30801': 'Septoplasty/Turbinectomy/Closed Nasal Bone Reduction',
    '30802': 'Septoplasty/Turbinectomy/Closed Nasal Bone Reduction',
    '30930': 'Septoplasty/Turbinectomy/Closed Nasal Bone Reduction',

    # Head and neck
    '38510': 'Head and Neck Procedures',
    '60220': 'Head and Neck Procedures',
    '60240': 'Head and Neck Procedures',
    '21556': 'Head and Neck Procedures',
    '42415': 'Head and Neck Procedures',
    '42440': 'Head and Neck Procedures',

    # Rhinologic
    '30117': 'Rhinologic Procedures',
    '30901': 'Rhinologic Procedures',
    '31237': 'Rhinologic Procedures',
    '31238': 'Rhinologic Procedures',
    '31256': 'Rhinologic Procedures',
    '31267': 'Rhinologic Procedures',

    # Airway
    '40819': 'Airway Procedures',
    '31545': 'Airway Procedures',
    '40806': 'Airway Procedures',
}

def filter_solo_cases(data: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only single-procedure HCUP cases.
    """
    solo = data[data["NCPT"] == 1].copy()
    print(f"{len(solo)} single-procedure cases remaining")
    return solo

def ttest_optime_by_cpt(data: pd.DataFrame, reference_csv: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Conducts a t-test for each CPT code with bonferroni correction
    """
    ref = pd.read_csv(reference_csv)
    ref['CPT'] = standardize_cpt(ref['CPT Code'])
    ref = ref.set_index('CPT')['Intra Time']

    solo = filter_solo_cases(data).dropna(subset=['ORTIME'])
    solo['CPT'] = standardize_cpt(solo['CPT1'])

    solo['CPT GROUP'] = solo['CPT'].map(group_map)
    solo = solo.dropna(subset=['CPT GROUP'])

    solo = solo.dropna(thresh=10, axis=1)
    cpts = sorted(solo['CPT'].unique())

    results = []
    for cpt in cpts:
        if cpt not in ref.index:
            print(f"CPT {cpt}: no reference mean found — skipping")
            continue

        group = solo[solo['CPT'] == cpt]['ORTIME']
        ref_mean = ref[cpt]

        if len(group) < 2:
            print(f"CPT {cpt}: not enough cases ({len(group)}) — skipping")
            continue

        t_stat, p_val = stats.ttest_1samp(group, popmean=ref_mean, alternative='two-sided')

        results.append({
            'CPT':            cpt,
            'n':              len(group),
            'observed_mean':  group.mean(),
            'observed_std':   group.std(),
            'reference_mean': ref_mean,
            'mean_diff':      group.mean() - ref_mean,
            't_stat':         t_stat,
            'p_value':        p_val,
            'significant':    p_val < alpha,
        })

    results_df = pd.DataFrame(results).sort_values('CPT')

    # Bonferroni correction for multiple comparisons
    n_tests = len(results_df)
    results_df['p_bonferroni'] = (results_df['p_value'] * n_tests).clip(upper=1.0)
    results_df['significant_bonferroni'] = results_df['p_bonferroni'] < alpha

    print(f"{results_df['significant'].sum()}/{len(results_df)} codes significant at p<{alpha}")
    print(f"{results_df['significant_bonferroni'].sum()}/{len(results_df)} codes significant after Bonferroni correction")
    results_df.to_csv("prelim_29_stats_results_optime_hcup.csv", index=False)
    return results_df

def plot_optime_boxplots(data: pd.DataFrame, reference_csv: str, results_df: pd.DataFrame = None):
    """
    Boxplot of mean operative time compared to reference mean for each CPT code, given the csv
    of statistics testing results
    """
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.major.width'] = 2.5
    plt.rcParams['ytick.major.width'] = 2.5
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['grid.linewidth'] = 1.5

    ref = pd.read_csv(reference_csv)
    ref['CPT'] = standardize_cpt(ref['CPT Code'])
    ref = ref.set_index('CPT')['Intra Time']

    solo = filter_solo_cases(data).dropna(subset=['ORTIME'])
    solo['CPT'] = standardize_cpt(solo['CPT1'])
    solo['CPT GROUP'] = solo['CPT'].map(group_map)
    solo = solo.dropna(subset=['CPT GROUP']) # shortened ver sina asked for hcup

    if results_df is not None:
        valid_cpts = set(standardize_cpt(results_df['CPT']))
        solo = solo[solo['CPT'].isin(valid_cpts)]

    solo = solo.dropna(thresh=10, axis=1).sort_values(by='CPT')

    group_order = sorted(solo['CPT GROUP'].dropna().unique())
    cpt_order = (
        solo[['CPT', 'CPT GROUP']]
        .drop_duplicates()
        .sort_values(['CPT GROUP', 'CPT'])['CPT']
        .tolist()
    )

    n_groups = len(group_order)
    n_cols = 2
    n_rows = -(-n_groups // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, n_rows * 9))
    axes = axes.flatten()

    sig_lookup = {}
    diff_lookup = {}
    n_lookup = {}
    if results_df is not None:
        rdf = results_df.copy()
        rdf['CPT'] = standardize_cpt(rdf['CPT'])
        sig_lookup  = rdf.set_index('CPT')['significant_bonferroni'].to_dict()
        diff_lookup = rdf.set_index('CPT')['mean_diff'].to_dict()
        n_lookup    = rdf.set_index('CPT')['n'].to_dict()

    for ax_idx, group_name in enumerate(group_order):
        ax = axes[ax_idx]
        chunk = [c for c in cpt_order if solo.loc[solo['CPT'] == c, 'CPT GROUP'].iloc[0] == group_name]

        plot_data = [solo[solo['CPT'] == cpt]['ORTIME'].values for cpt in chunk]

        bp = ax.boxplot(
            plot_data,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            labels=[str(c) for c in chunk],
            meanprops=dict(marker='^', markerfacecolor='green',
                           markeredgecolor='green', markersize=15),
            medianprops=dict(color='orange', linewidth=3),
            boxprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=2),
            capprops=dict(linewidth=2),
        )

        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.5)

        ax.autoscale()
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.12, y_max)
        y_min_new = y_min - y_range * 0.12

        for cpt_idx, cpt in enumerate(chunk):
            x_pos = cpt_idx + 1

            if cpt in ref.index:
                ref_mean = ref[cpt]
                ax.plot(
                    [x_pos - 0.4, x_pos + 0.4], [ref_mean, ref_mean],
                    color='red', linewidth=2.5, linestyle='--', zorder=5,
                    label='RUC reference mean' if cpt_idx == 0 else None
                )

            if cpt in diff_lookup:
                diff = diff_lookup[cpt]
                sign = '+' if diff >= 0 else ''
                ax.annotate(
                    f'{sign}{diff:.0f}m',
                    xy=(x_pos, y_min_new + y_range * 0.02),
                    ha='center', va='bottom', fontsize=18,
                    color='darkgreen' if sig_lookup.get(cpt, True) else 'firebrick',
                    annotation_clip=False
                )

        x_labels = []
        for cpt in chunk:
            star = '*' if sig_lookup.get(cpt, False) else ''
            x_labels.append(f'{cpt}{star}\nN = {n_lookup.get(cpt, "")}')

        ax.set_xticklabels(x_labels, fontsize=18, linespacing = 1.4)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_title(f'Group: {group_name}', fontsize=24, fontweight='bold', pad = 12)
        ax.set_ylabel('Operative Time (min)', fontsize=24, labelpad = 10)
        ax.set_xlabel('CPT Code', fontsize=24, labelpad = 18)
        ax.grid(True, alpha=0.3, axis='y')

    for ax in axes[n_groups:]:
        ax.set_visible(False)

    legend_elements = [
    Line2D([0], [0], color='red', linewidth=2.5, linestyle='--', label='RUC reference mean'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
           markersize=11, label='Observed mean'),
    Line2D([0], [0], color='orange', linewidth=2.5, label='Observed median'),
    Patch(facecolor='steelblue', alpha=0.5, label='IQR (25th–75th pct)'),
    ]
    fig.legend(handles=legend_elements, fontsize=20, loc='lower center',
            ncol=4, borderpad=0.6)

    # and update tight_layout to leave room at the bottom for it
    plt.tight_layout(rect=[0, 0.04, 1, 0.93]) # 0.93

    fig.suptitle(
        'Operative Time by CPT Code (solo cases only)\n* = significant after Bonferroni correction', fontsize=30, fontweight='bold'
    )
    plt.savefig('optime_boxplots_cptgrouped.png', dpi=300, bbox_inches='tight')
    plt.savefig('optime_boxplots_grouped.svg', dpi=300, bbox_inches='tight', format="svg")
    plt.show()

def plot_optime_boxplots_poster(
    data: pd.DataFrame,
    reference_csv: str,
    results_df: pd.DataFrame,
    top_n: int = 5,
    figsize: tuple = (8, 6) # changed to 10, 7
):
    """
    Plots 5 codes, for poster purposes only
    """
    ref = pd.read_csv(reference_csv)
    ref['CPT'] = standardize_cpt(ref['CPT Code'])
    ref = ref.set_index('CPT')['Intra Time']

    solo = filter_solo_cases(data).dropna(subset=['ORTIME'])
    solo['CPT'] = standardize_cpt(solo['CPT1'])
    solo['CPT GROUP'] = solo['CPT'].map(group_map)

    rdf = results_df.copy()
    rdf['CPT'] = standardize_cpt(rdf['CPT'])
    rdf['abs_diff'] = rdf['mean_diff'].abs()
    sig_lookup  = rdf.set_index('CPT')['significant_bonferroni'].to_dict()
    diff_lookup = rdf.set_index('CPT')['mean_diff'].to_dict()
    n_lookup    = rdf.set_index('CPT')['n'].to_dict()

    top_cpts = (
        rdf[rdf['significant_bonferroni']]
        .sort_values('abs_diff', ascending=False)
        .head(top_n)['CPT']
        .tolist()
    )

    solo = solo[solo['CPT'].isin(top_cpts)]

    group_lookup = (
        solo[['CPT', 'CPT GROUP']]
        .drop_duplicates()
        .set_index('CPT')['CPT GROUP']
        .to_dict()
    )

    group_short = {
        'Diagnostic Procedures': 'Diagnostic',
        'In-Office Procedures': 'Office',
        'Septoplasty/Turbinectomy/Closed Nasal Bone Reduction': 'Turbinate',
        'Head and Neck Procedures': 'Head & Neck',
        'Rhinologic Procedures': 'Rhinologic',
        'Airway Procedures': 'Airway',
    }

    chunk = sorted(top_cpts, key=lambda c: abs(diff_lookup.get(c, 0)), reverse=True)
    plot_data = [solo[solo['CPT'] == cpt]['ORTIME'].values for cpt in chunk]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        plot_data,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        vert=True,
        meanprops=dict(marker='^', markerfacecolor='green',
                       markeredgecolor='green', markersize=7), # changed to 10
        medianprops=dict(color='orange', linewidth=2), # changed to 2.5
    )

    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.5)

    # clip y axis at 95th percentile across all plotted data to reduce whitespace
    ax.autoscale()
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    for cpt_idx, cpt in enumerate(chunk):
        x_pos = cpt_idx + 1

        if cpt in ref.index:
            ax.plot(
                [x_pos - 0.4, x_pos + 0.4], [ref[cpt], ref[cpt]],
                color='red', linewidth=2, linestyle='--', zorder=5
            )

        if cpt in diff_lookup:
            diff = diff_lookup[cpt]
            sign = '+' if diff >= 0 else ''
            ax.annotate(
                f'{sign}{diff:.0f}m',
                xy=(x_pos, y_min + y_range * 0),
                ha='center', va='bottom', fontsize=8, # changed to 10
                color='darkgreen' if diff > 0 else 'firebrick',
                annotation_clip=False
            )

    # compact x labels: CPT + star, n, short group — each on own line
    x_labels = []
    for cpt in chunk:
        star = '*' if sig_lookup.get(cpt, False) else ''
        group = group_short.get(group_lookup.get(cpt, ''), group_lookup.get(cpt, ''))
        n = n_lookup.get(cpt, '')
        x_labels.append(f'{cpt}{star}\nn={n}\n{group}')

    ax.set_xticklabels(x_labels, fontsize=10) # changed to 16
    #ax.tick_params(axis='y', labelsize=14) # changed to 14
    ax.set_ylabel('Operative Time (min)', fontsize=18) # changed to 16
    ax.set_xlabel('')
    ax.set_title( 
        'HCUP Top 5 CPTs:\nOperative Time vs.\n RUC Reference', # changed to two lines
        fontsize=20, fontweight='bold' # and to 22
    )
    ax.grid(True, alpha=0.3, axis='y')

    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='RUC mean'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               markersize=8, label='Observed mean'),
        Line2D([0], [0], color='orange', linewidth=2, label='Median'), 
        Patch(facecolor='steelblue', alpha=0.5, label='IQR'),
    ]
    fig.legend(handles=legend_elements, fontsize=7, loc='lower center', # changed to fontsize 11
               ncol=2, bbox_to_anchor=(0.5, 0.0), borderpad=0.5) # changed anchor to 0.0 -> 0.01

    plt.tight_layout(rect=[0, 0.06, 1, 1]) # 0.06 -> 0.08
    plt.savefig('optime_boxplots_poster_hcup.png', dpi=300, bbox_inches='tight')
    plt.savefig('optime_boxplots_poster_hcup.svg', bbox_inches='tight')
    plt.show()

def plot_optime_linreg(data: pd.DataFrame, min_years: int = 5):
    """
    Plots linear regression for operative time trends over the years for each CPT code.
    """
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.major.width'] = 2.5
    plt.rcParams['ytick.major.width'] = 2.5
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['grid.linewidth'] = 1.5

    solo = filter_solo_cases(data).dropna(subset=['ORTIME', 'AYEAR'])
    solo['CPT'] = standardize_cpt(solo['CPT1'])

    solo['CPT GROUP'] = solo['CPT'].map(group_map)
    solo = solo.dropna(subset=['CPT GROUP']) # drops the ones not in the new mapping
    group_order = sorted(solo['CPT GROUP'].dropna().unique())
    cpt_order = (
        solo[['CPT', 'CPT GROUP']]
        .drop_duplicates()
        .sort_values(['CPT GROUP', 'CPT'])
    )

    n_cols = 2
    n_rows = -(-len(group_order) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, n_rows * 8))
    axes = axes.flatten()

    for ax_idx, group_name in enumerate(group_order):
        ax = axes[ax_idx]
        color_cycle = plt.cm.tab10.colors

        chunk = cpt_order[cpt_order['CPT GROUP'] == group_name]['CPT'].tolist()

        for color_idx, cpt in enumerate(chunk):
            color = color_cycle[color_idx % len(color_cycle)]
            df_cpt = solo[solo['CPT'] == cpt]

            df_grouped = (
                df_cpt.groupby('AYEAR')['ORTIME']
                .mean()
                .reset_index(name='AverageOperativeTime')
                .sort_values('AYEAR')
            )

            if len(df_grouped) < min_years:
                print(f"CPT {cpt}: skipped (only {len(df_grouped)} years of data)")
                continue

            X = df_grouped['AYEAR'].values.reshape(-1, 1)
            y = df_grouped['AverageOperativeTime'].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r_squared = model.score(X, y)

            print(f"CPT {cpt}: slope={model.coef_[0]:.2f} min/year, R²={r_squared:.2f}")

           
            plt.figure(figsize=(8, 6)) 
            residuals = y - y_pred
            plt.scatter(y_pred, residuals, color='blue', alpha=0.6)
            plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
            plt.xlabel('Predicted Operative Time (min)', fontsize=14)
            plt.ylabel('Residuals (Actual - Predicted)', fontsize=14)
            plt.title(f'Linearity Check: Residuals for CPT {cpt}', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.show()
        
            # actual data: dots connected by solid line
            ax.plot(
                df_grouped['AYEAR'],
                y,
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=5,
                alpha=0.7,
                zorder=2
            )

            # regression: dashed line + label
            ax.plot(
                df_grouped['AYEAR'],
                y_pred,
                color=color,
                linewidth=3,
                linestyle='--',
                label=f'{cpt} (slope={model.coef_[0]:.2f}, R²={r_squared:.2f})',
                zorder=3
            )

        all_years = sorted(solo['AYEAR'].unique())
        ax.set_xticks(all_years[::2])
        ax.set_xticklabels([str(y) for y in all_years[::2]], fontsize=18)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_title(f'Group: {group_name}', fontsize=22, fontweight='bold', pad=10)
        ax.set_xlabel('Year', fontsize=20, labelpad = 8)
        ax.set_ylabel('Avg Operative Time (min)', fontsize=20, labelpad = 8)
        ax.legend(
                fontsize=16, loc='upper left',
                framealpha=0.9,
                borderpad=0.4,
                labelspacing=0.3,
            )
        ax.grid(True, alpha=0.3)

    for ax in axes[len(group_order):]:
        ax.set_visible(False)

    fig.suptitle(
        'Trend of Average Operative Time by CPT Code (solo cases)\nDashed = linear regression, Solid = observed mean',
        fontsize=30, fontweight = 'bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('optime_linreg_subplots_cptgrouped.png', dpi=300, bbox_inches='tight')
    plt.savefig('optime_linreg_subplots_cptgrouped.svg',  bbox_inches='tight')
    plt.show()

def main():
    df = pd.read_csv("HCUP_filtered_172.csv") # filtered file from filtering.py
    #resultsdf = ttest_optime_by_cpt(df, "filtered_sina2.csv") # csv with RUC reference means
    #plot_optime_boxplots(df, "filtered_sina2.csv", resultsdf)
    #plot_optime_boxplots_poster(df, "filtered_sina2.csv", resultsdf)
    plot_optime_linreg(df)

if __name__ == "__main__":
    main()