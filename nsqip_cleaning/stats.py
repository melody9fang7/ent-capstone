from scipy import stats
import pandas as pd
import numpy as np
from data_handling_nsqip import standardize_cpt
import matplotlib.pyplot as plt

def filter_solo_cases(data: pd.DataFrame) -> pd.DataFrame:
    other_cols = [c for c in data.columns if c.startswith('OTHERCPT')]
    is_solo = data[other_cols].isnull().all(axis=1)
    n_removed = (~is_solo).sum()
    print(f"Removed {n_removed} non-solo cases ({n_removed/len(data)*100:.1f}%), {is_solo.sum()} remaining")
    return data[is_solo]


def ttest_optime_by_cpt(data: pd.DataFrame, reference_csv: str, alpha: float = 0.05) -> pd.DataFrame:
    ref = pd.read_csv(reference_csv)
    ref['CPT'] = standardize_cpt(ref['CPT'])
    ref = ref.set_index('CPT')['Intra Time']  # CPT -> reference mean
    
    solo = filter_solo_cases(data)
    solo['CPT'] = standardize_cpt(solo['CPT'])
    solo = solo.dropna(subset=['OPTIME'])

    results = []
    for cpt in sorted(solo['CPT'].unique()):
        if cpt not in ref.index:
            print(f"  CPT {cpt}: no reference mean found — skipping")
            continue

        group = solo[solo['CPT'] == cpt]['OPTIME']
        ref_mean = ref[cpt]

        if len(group) < 2:
            print(f"  CPT {cpt}: not enough cases ({len(group)}) — skipping")
            continue

        t_stat, p_val = stats.ttest_1samp(group, popmean=ref_mean,  alternative='greater')

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

    results_df = pd.DataFrame(results).sort_values('p_value')

    # Bonferroni correction for multiple comparisons
    n_tests = len(results_df)
    results_df['p_bonferroni'] = (results_df['p_value'] * n_tests).clip(upper=1.0)
    results_df['significant_bonferroni'] = results_df['p_bonferroni'] < alpha

    print(f"{results_df['significant'].sum()}/{len(results_df)} codes significant at p<{alpha}")
    print(f"{results_df['significant_bonferroni'].sum()}/{len(results_df)} codes significant after Bonferroni correction")
    results_df.to_csv("prelim_29_stats_results_optime.csv")
    return results_df

def plot_optime_boxplots(data: pd.DataFrame, reference_csv: str, results_df: pd.DataFrame = None):
    """
    Boxplot of solo operative time for each CPT code, with reference mean marked.
    If results_df is provided, marks significant codes with * in the title.
    """
    ref = pd.read_csv(reference_csv)
    ref['CPT'] = standardize_cpt(ref['CPT']) 
    ref = ref.set_index('CPT')['Intra Time']

    solo = filter_solo_cases(data).dropna(subset=['OPTIME'])
    solo['CPT'] = standardize_cpt(solo['CPT']) 
    cpts = sorted(solo['CPT'].unique())

    chunks = [cpts[i:i+5] for i in range(0, len(cpts), 5)]
    n_cols = 2
    n_rows = -(-len(chunks) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
    axes = axes.flatten()

    for ax_idx, chunk in enumerate(chunks):
        ax = axes[ax_idx]

        plot_data = [solo[solo['CPT'] == cpt]['OPTIME'].values for cpt in chunk]
        bp = ax.boxplot(plot_data, patch_artist=True,showmeans=True, showfliers=False,
                        labels=[str(c) for c in chunk])

        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)

        for cpt_idx, cpt in enumerate(chunk):
            if cpt not in ref.index:
                continue
            ref_mean = ref[cpt]
            # draw reference mean as a horizontal line across that box's x position
            ax.plot([cpt_idx + 0.6, cpt_idx + 1.4], [ref_mean, ref_mean],
                    color='red', linewidth=2, linestyle='--', zorder=5,
                    label='Reference mean' if cpt_idx == 0 else None)

        # mark significant codes with * if results provideda
        if results_df is not None:
            sig = results_df.set_index('CPT')['significant_bonferroni']
            labels = []
            for cpt in chunk:
                star = '*' if (cpt in sig.index and sig[cpt]) else ''
                labels.append(f'{cpt}{star}')
            ax.set_xticklabels(labels, fontsize=8)

        ax.set_ylabel('Operative Time (min)', fontsize=9)
        ax.set_xlabel('CPT Code', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    for ax in axes[len(chunks):]:
        ax.set_visible(False)

    fig.suptitle('Operative Time by CPT Code (solo cases only)\n* = significant after Bonferroni correction',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('figs/optime_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    df = pd.read_csv("C:/Users/melod/Desktop/prog/170a/proj/ent-capstone/data/nsqip/combined_filtered_29.csv")
    resultsdf = ttest_optime_by_cpt(df, "C:/Users/melod/Desktop/prog/170a/proj/ent-capstone/data/final_CPT_1.csv")

    plot_optime_boxplots(df, "C:/Users/melod/Desktop/prog/170a/proj/ent-capstone/data/final_CPT_1.csv", resultsdf)

if __name__ == "__main__":
    main()