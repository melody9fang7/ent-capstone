from scipy import stats
import pandas as pd
import numpy as np
from data_handling_nsqip import standardize_cpt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def filter_solo_cases(data: pd.DataFrame) -> pd.DataFrame:
    other_cols = [c for c in data.columns if c.startswith('OTHERCPT')]
    is_solo = data[other_cols].isnull().all(axis=1)
    n_removed = (~is_solo).sum()
    print(f"Removed {n_removed} non-solo cases ({n_removed/len(data)*100:.1f}%), {is_solo.sum()} remaining")
    return data[is_solo]


def ttest_optime_by_cpt(data: pd.DataFrame, reference_csv: str, alpha: float = 0.05) -> pd.DataFrame:
    ref = pd.read_csv(reference_csv)
    ref['CPT'] = standardize_cpt(ref['CPT']) 
    ref = ref.set_index('CPT')['Intra Time']

    solo = filter_solo_cases(data).dropna(subset=['OPTIME'])
    solo['CPT'] = standardize_cpt(solo['CPT'])
    solo = solo.dropna(thresh=10, axis=1)

    groups = sorted(solo['CPT GROUP'].dropna().unique())

    cpts = sorted(solo['CPT'].unique())

    results = []
    for cpt in cpts:
        if cpt not in ref.index:
            print(f"  CPT {cpt}: no reference mean found — skipping")
            continue

        group = solo[solo['CPT'] == cpt]['OPTIME']
        ref_mean = ref[cpt]

        if len(group) < 2:
            print(f"  CPT {cpt}: not enough cases ({len(group)}) — skipping")
            continue

        t_stat, p_val = stats.ttest_1samp(group, popmean=ref_mean,  alternative='two-sided')

        results.append({
            'CPT':           cpt,
            'n':             len(group),
            'observed_mean': group.mean(),
            'observed_std':  group.std(),
            'reference_mean': ref_mean,
            'mean_diff':     group.mean() - ref_mean,
            't_stat':        t_stat,
            'p_value':       p_val,
            'significant':   p_val < alpha,
        })

    results_df = pd.DataFrame(results).sort_values('CPT')

    # Bonferroni correction for multiple comparisons
    n_tests = len(results_df)
    results_df['p_bonferroni'] = (results_df['p_value'] * n_tests).clip(upper=1.0)
    results_df['significant_bonferroni'] = results_df['p_bonferroni'] < alpha

    print(f"{results_df['significant'].sum()}/{len(results_df)} codes significant at p<{alpha}")
    print(f"{results_df['significant_bonferroni'].sum()}/{len(results_df)} codes significant after Bonferroni correction")
    results_df.to_csv("prelim_29_stats_results_optime_hcup.csv")
    return results_df

def plot_optime_boxplots(data: pd.DataFrame, reference_csv: str, results_df: pd.DataFrame = None):
    """
    boxplot of solo operative time for each CPT code, with reference mean marked.
    if results_df is provided, marks significant codes with * in the title.
    """
    ref = pd.read_csv(reference_csv)
    ref['CPT'] = standardize_cpt(ref['CPT'])
    ref = ref.set_index('CPT')['Intra Time']

    solo = filter_solo_cases(data).dropna(subset=['OPTIME'])
    solo['CPT'] = standardize_cpt(solo['CPT'])

    if results_df is not None:
        valid_cpts = set(standardize_cpt(results_df['CPT']))
        solo = solo[solo['CPT'].isin(valid_cpts)]

    solo = solo.dropna(thresh=10, axis=1).sort_values(by='CPT')

    group_order = sorted(solo['CPT GROUP'].unique())
    cpt_order = (
        solo[['CPT', 'CPT GROUP']]
        .drop_duplicates()
        .sort_values(['CPT GROUP', 'CPT'])['CPT']
        .tolist()
    )

    n_groups = len(group_order)
    n_cols = 2
    n_rows = -(-n_groups // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 7))
    axes = axes.flatten()

    sig_lookup = {}
    diff_lookup = {}
    n_lookup = {}
    if results_df is not None:
        rdf = results_df.copy()
        rdf['CPT'] = standardize_cpt(rdf['CPT'])
        sig_lookup  = rdf.set_index('CPT')['significant_bonferroni'].to_dict()
        diff_lookup = rdf.set_index('CPT')['mean_diff'].to_dict()
        n_lookup = rdf.set_index('CPT')['n'].to_dict()

    for ax_idx, group_name in enumerate(group_order):
        ax = axes[ax_idx]
        chunk = [c for c in cpt_order if solo.loc[solo['CPT'] == c, 'CPT GROUP'].iloc[0] == group_name]

        plot_data = [solo[solo['CPT'] == cpt]['OPTIME'].values for cpt in chunk]

        bp = ax.boxplot(
            plot_data,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            labels=[str(c) for c in chunk],
            meanprops=dict(marker='^', markerfacecolor='green',
                           markeredgecolor='green', markersize=8),
            medianprops=dict(color='orange', linewidth=2),
        )

        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.5)
            
        ax.autoscale()
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        for cpt_idx, cpt in enumerate(chunk):
            x_pos = cpt_idx + 1

            if cpt in ref.index:
                ref_mean = ref[cpt]
                ax.plot(
                    [x_pos - 0.4, x_pos + 0.4], [ref_mean, ref_mean],
                    color='red', linewidth=2, linestyle='--', zorder=5,
                    label='RUC reference mean' if cpt_idx == 0 else None
                )

            if cpt in diff_lookup:
                diff = diff_lookup[cpt]
                sign = '+' if diff >= 0 else ''
                ax.annotate(
                    f'{sign}{diff:.0f}m',
                    xy=(x_pos, y_min + y_range * 0.02),  # just above the bottom
                    ha='center', va='bottom', fontsize=7,
                    color='darkgreen' if sig_lookup.get(cpt, True)  else 'firebrick',
                    annotation_clip=False
                )

        x_labels = []

        for cpt_idx, cpt in enumerate(chunk):
            star = '*' if sig_lookup.get(cpt, False) else ''
            x_labels.append(f'{cpt}{star}\nN = {n_lookup.get(cpt, False)}')

        ax.set_xticklabels(x_labels, fontsize=9)

        ax.set_title(f'Group: {group_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Operative Time (min)', fontsize=10)
        ax.set_xlabel('CPT Code', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    for ax in axes[n_groups:]:
        ax.set_visible(False)

    legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='RUC reference mean'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
           markersize=9, label='Observed mean'),
    Line2D([0], [0], color='orange', linewidth=2, label='Observed median'),
    Patch(facecolor='steelblue', alpha=0.5, label='IQR (25th–75th pct)'),
    ]
    fig.legend(handles=legend_elements, fontsize=9, loc='lower center',
            ncol=4, bbox_to_anchor=(0.5, 0.0), borderpad=0.5)

    # and update tight_layout to leave room at the bottom for it
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    fig.suptitle(
        'Operative Time by CPT Code (solo cases only)\n* = significant after Bonferroni correction', fontsize=16, fontweight='bold'
    )
    plt.savefig('finalfigs/optime_boxplots_cptgrouped.png', dpi=300, bbox_inches='tight')
    plt.savefig('finalfigs/optime_boxplots_cptgrouped.svg',  bbox_inches='tight')
    plt.show()

def plot_optime_boxplots_poster(
    data: pd.DataFrame,
    reference_csv: str,
    results_df: pd.DataFrame,
    top_n: int = 5,
    figsize: tuple = (8, 5)
):
    ref = pd.read_csv(reference_csv)
    ref['CPT'] = standardize_cpt(ref['CPT'])
    ref = ref.set_index('CPT')['Intra Time']

    solo = filter_solo_cases(data).dropna(subset=['OPTIME'])
    solo['CPT'] = standardize_cpt(solo['CPT'])

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

    # shorten group names so they fit
    group_short = {
        'Glossectomies and Laryngectomies': 'Gloss. & Laryngect.',
        'Other Oral Cavity Resections': 'Oral Cavity',
        'Thyroid Surgeries': 'Thyroid',
        'Neck Dissections': 'Neck Dissection',
        'Salivary Gland Surgeries': 'Salivary Gland',
        'Miscellaneous Codes': 'Misc.',
    }

    chunk = sorted(top_cpts, key=lambda c: abs(diff_lookup.get(c, 0)), reverse=True)
    plot_data = [solo[solo['CPT'] == cpt]['OPTIME'].values for cpt in chunk]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        plot_data,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        vert=True,
        meanprops=dict(marker='^', markerfacecolor='green',
                       markeredgecolor='green', markersize=7),
        medianprops=dict(color='orange', linewidth=2),
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
                xy=(x_pos, y_min + y_range * 0.02),
                ha='center', va='bottom', fontsize=8,
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

    ax.set_xticklabels(x_labels, fontsize=7.5)
    ax.set_ylabel('Operative Time (min)', fontsize=9)
    ax.set_xlabel('')
    ax.set_title(
        'NSQIP Top 5 CPTs:\nOperative Time vs.\nRUC Reference',
        fontsize=10, fontweight='bold' 
    )
    ax.grid(True, alpha=0.3, axis='y')

    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='RUC mean'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               markersize=8, label='Observed mean'),
        Line2D([0], [0], color='orange', linewidth=2, label='Median'),
        Patch(facecolor='steelblue', alpha=0.5, label='IQR'),
    ]
    fig.legend(handles=legend_elements, fontsize=7, loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, 0.0), borderpad=0.4)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('finalfigs/optime_boxplots_poster.png', dpi=300, bbox_inches='tight')
    plt.savefig('finalfigs/optime_boxplots_poster.svg', bbox_inches='tight')
    plt.show()

def plot_optime_linreg(data: pd.DataFrame, min_years: int = 5):
    solo = filter_solo_cases(data).dropna(subset=['OPTIME', 'PUFYEAR'])
    solo['CPT'] = standardize_cpt(solo['CPT'])

    group_order = sorted(solo['CPT GROUP'].dropna().unique())
    cpt_order = (
        solo[['CPT', 'CPT GROUP']]
        .drop_duplicates()
        .sort_values(['CPT GROUP', 'CPT'])
    )

    n_cols = 2
    n_rows = -(-len(group_order) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    axes = axes.flatten()

    for ax_idx, group_name in enumerate(group_order):
        ax = axes[ax_idx]
        color_cycle = plt.cm.tab10.colors

        chunk = cpt_order[cpt_order['CPT GROUP'] == group_name]['CPT'].tolist()

        for color_idx, cpt in enumerate(chunk):
            color = color_cycle[color_idx % len(color_cycle)]
            df_cpt = solo[solo['CPT'] == cpt]

            df_grouped = (
                df_cpt.groupby('PUFYEAR')['OPTIME']
                .mean()
                .reset_index(name='AverageOperativeTime')
                .sort_values('PUFYEAR')
            )

            if len(df_grouped) < min_years:
                print(f"CPT {cpt}: skipped (only {len(df_grouped)} years of data)")
                continue

            X = df_grouped['PUFYEAR'].values.reshape(-1, 1)
            y = df_grouped['AverageOperativeTime'].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r_squared = model.score(X, y)

            print(f"CPT {cpt}: slope={model.coef_[0]:.2f} min/year, R²={r_squared:.2f}")

            # actual data: dots connected by solid line
            ax.plot(
                df_grouped['PUFYEAR'],
                y,
                color=color,
                linewidth=1.5,
                marker='o',
                markersize=4,
                alpha=0.7,
                zorder=2
            )

            # regression: dashed line + label
            ax.plot(
                df_grouped['PUFYEAR'],
                y_pred,
                color=color,
                linewidth=2,
                linestyle='--',
                label=f'{cpt} (slope={model.coef_[0]:.2f}, R²={r_squared:.2f})',
                zorder=3
            )

        ax.set_title(f'Group: {group_name}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Avg Operative Time (min)', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    for ax in axes[len(group_order):]:
        ax.set_visible(False)

    fig.suptitle(
        'Trend of Average Operative Time by CPT Code (solo cases)\nDashed = linear regression, Solid = observed mean',
        fontsize=13
    )

    plt.tight_layout()
    plt.savefig('finalfigs/optime_linreg_subplots_cptgrouped.png', dpi=300, bbox_inches='tight')
    plt.savefig('finalfigs/optime_linreg_subplots_cptgrouped.svg',  bbox_inches='tight')
    plt.show()

def main():
    df = pd.read_csv("C:/Users/melod/Desktop/prog/170a/proj/ent-capstone/data/nsqip/combined_filtered.csv")
    resultsdf = ttest_optime_by_cpt(df, "C:/Users/melod/Desktop/prog/170a/proj/ent-capstone/data/final_CPT_1.csv")
    #plot_optime_linreg(df, 0)
    #plot_optime_boxplots(df, "C:/Users/melod/Desktop/prog/170a/proj/ent-capstone/data/final_CPT_1.csv", resultsdf)
    plot_optime_boxplots_poster(df, "C:/Users/melod/Desktop/prog/170a/proj/ent-capstone/data/final_CPT_1.csv", resultsdf)
if __name__ == "__main__":
    main()