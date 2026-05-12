from scipy import stats
import pandas as pd
import numpy as np
from data_handling_nsqip import standardize_cpt
import matplotlib.pyplot as plt

gloss_and_laryn = [31360, 31365, 41120, 41130, 41135, 41155]  # Glossectomies and Laryngectomies
oral_cav = [21044, 40810, 40816, 42120, 42842]  # Other Oral Cavity Resections
neck_diss = [38542, 38700, 38720, 38724]  # Neck Dissections
salivary_gland = [42415, 42420, 42440]  # Salivary Gland Surgeries
thyroid = [60220, 60240, 60252, 60254, 60260, 60270, 60271]  # Thyroid Surgeries
misc_codes = [15731, 21556, 31591, 42145]  # Miscellaneous Codes

CLASSES = {
    'Glossectomy_Laryngectomy': [str(c) for c in gloss_and_laryn],
    'Oral_Cavity': [str(c) for c in oral_cav],
    'Neck_Dissection': [str(c) for c in neck_diss],
    'Salivary_Gland': [str(c) for c in salivary_gland],
    'Thyroid': [str(c) for c in thyroid],
    'Miscellaneous': [str(c) for c in misc_codes]
}


def filter_solo_cases(data: pd.DataFrame) -> pd.DataFrame:
    other_cols = [c for c in data.columns if c.startswith('OTHERCPT')]
    is_solo = data[other_cols].isnull().all(axis=1)
    n_removed = (~is_solo).sum()
    print(f"Removed {n_removed} non-solo cases ({n_removed/len(data)*100:.1f}%), {is_solo.sum()} remaining")
    return data[is_solo].copy()


def ttest_optime_by_cpt(data: pd.DataFrame, reference_csv: str, alpha: float = 0.05) -> pd.DataFrame:
    ref = pd.read_csv(reference_csv)
    ref['CPT'] = standardize_cpt(ref['CPT'])
    ref = ref.set_index('CPT')['Intra Time']
    
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

        t_stat, p_val = stats.ttest_1samp(group, popmean=ref_mean, alternative='greater')

        results.append({
            'CPT': cpt,
            'n': len(group),
            'observed_mean': group.mean(),
            'observed_std': group.std(),
            'reference_mean': ref_mean,
            'mean_diff': group.mean() - ref_mean,
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < alpha,
        })

    results_df = pd.DataFrame(results).sort_values('p_value')

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
    ** Also now grouping by CPT class
    """
    ref = pd.read_csv(reference_csv)
    ref['CPT'] = standardize_cpt(ref['CPT']) 
    ref = ref.set_index('CPT')['Intra Time']

    solo = filter_solo_cases(data).dropna(subset=['OPTIME'])
    solo['CPT'] = standardize_cpt(solo['CPT']) 
    cpts = sorted(solo['CPT'].unique())

    for class_name, class_cpts in CLASSES.items():
        available_cpts = [cpt for cpt in class_cpts if cpt in solo['CPT'].unique()]
                
        if len(available_cpts) == 0:
            print(f"\tNo data for {class_name}")
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_data = [solo[solo['CPT'] == cpt]['OPTIME'].values for cpt in available_cpts]
        labels = [cpt for cpt in available_cpts]
        
        bp = ax.boxplot(plot_data, patch_artist=True, showmeans=True, showfliers=False,
                        labels=labels)
        
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)        

        for cpt_idx, cpt in enumerate(available_cpts):
            cpt_int = int(cpt) if cpt.isdigit() else cpt
            if cpt_int in ref.index:
                ref_mean = ref[cpt_int]
                # draw reference mean as a horizontal line across that box's x position
                ax.plot([cpt_idx + 0.6, cpt_idx + 1.4], [ref_mean, ref_mean],
                        color='red', linewidth=2, linestyle='--', zorder=5,
                        label='Reference mean' if cpt_idx == 0 else None)

            elif cpt in ref.index: 
                ref_mean = ref[cpt]
                ax.plot([cpt_idx + 0.6, cpt_idx + 1.4], [ref_mean, ref_mean],
                        color='red', linewidth=2, linestyle='--', zorder=5,
                        label='Reference mean' if cpt_idx == 0 else None)
        
        # mark significant codes with * if results provideda
        if results_df is not None:
            sig = results_df.set_index('CPT')['significant_bonferroni']
            new_labels = []
            for cpt in available_cpts:
                star = '*' if (cpt in sig.index and sig[cpt]) else ''
                new_labels.append(f'{cpt}{star}')
            ax.set_xticklabels(new_labels, fontsize=10)
        
        ax.set_ylabel('Operative Time (min)', fontsize=12)
        ax.set_xlabel('CPT Code', fontsize=12)
        ax.set_title(f'{class_name.replace("_", " ")}: Operative Time by CPT Code (solo cases only)\n* = significant after Bonferroni correction',
                     fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'optime_boxplots_{class_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"  Saved: optime_boxplots_{class_name}.png")


def main():
    #df = pd.read_csv("nsqip_cleaning/combined_filtered_29.csv")
    #df = pd.read_csv("nsqip-pediatrics/ALL_NSQIP-P.csv")
    files = ['nsqip_cleaning/combined_filtered_29.csv', 'nsqip-pediatrics/ALL_NSQIP-P.csv']
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    resultsdf = ttest_optime_by_cpt(df, "nsqip_cleaning/reference_times.csv")
    plot_optime_boxplots(df, "nsqip_cleaning/reference_times.csv", resultsdf)


if __name__ == "__main__":
    main()