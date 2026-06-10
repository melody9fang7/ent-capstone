"""
Used to filter and find initial list of CPT codes that had
been attributed to ENT procedures in the NSQIP data.

Dependencies:
    pip install pandas pathlib os
"""
import pandas as pd
import os
from pathlib import Path
import glob


def standardize_cpt(series: pd.Series) -> pd.Series:
    """
    forces formatting of cpt codes
    """
    def _to_str(x):
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return str(x)
 
    return series.apply(_to_str)


def find_ent_cpt_counts(df: pd.DataFrame) -> pd.DataFrame:
    '''
    given the df for combined data, returns a df of unique codes that have been attributed to SURGSPEC ENT at least once,
    general counts within the file, counts where SURGSPEC==ENT, counts where only an ENT procedure was performed, and average
    amount of procedures performed if it isn't just 1.
    '''
    other_cols = [col for col in df.columns if col.startswith('OTHERCPT')]

    df['SURGSPEC'] = df['SURGSPEC'].astype(str)
    df['CPT'] = df['CPT'].astype(str)
    df['num_othercpt'] = df[other_cols].notna().sum(axis=1)

    ent_cpt_codes = (
        df[df['SURGSPEC'] == "Otolaryngology (ENT)"]['CPT']
        .dropna()
        .unique()
    )

    ent_df = df[df['CPT'].isin(ent_cpt_codes)]

    counts = ent_df['CPT'].value_counts().reset_index()
    counts.columns = ["CPT", "count"]

    exact_ent_counts = (
        ent_df[ent_df['SURGSPEC'] == "Otolaryngology (ENT)"]['CPT']
        .value_counts()
        .reset_index()
    )
    exact_ent_counts.columns = ["CPT", "exact_ENT_count"]

    ent_only_df = ent_df[ent_df['SURGSPEC'] == "Otolaryngology (ENT)"]

    # cases where just 1 ENT procedure was performed alone
    ent_alone_counts = (
        ent_only_df[ent_only_df['num_othercpt'] == 0]['CPT']
        .value_counts()
        .reset_index()
    )
    ent_alone_counts.columns = ["CPT", "ent_alone_count"]

    ent_avg_other_counts = (
        ent_only_df[ent_only_df['num_othercpt'] > 0]
        .groupby('CPT')['num_othercpt']
        .mean()
        .reset_index()
    )
    ent_avg_other_counts.columns = ["CPT", "ent_avg_othercpt_count"]

    counts = counts.merge(exact_ent_counts, on="CPT", how="left")
    counts = counts.merge(ent_alone_counts, on="CPT", how="left")
    counts = counts.merge(ent_avg_other_counts, on="CPT", how="left")

    counts['exact_ENT_count'] = counts['exact_ENT_count'].fillna(0).astype(int)
    counts["ent_alone_count"] = counts["ent_alone_count"].fillna(0).astype(int)
    counts["ent_avg_othercpt_count"] = counts["ent_avg_othercpt_count"].fillna(0)

    return counts
 
def get_cpts(in_directory: str, cpt_out_path: str) -> pd.DataFrame:
    """
    given the path to the directory of CSV files for each year, gets list of CPT codes where they've been attributed to ENT at least once
    and don't drop off after 2010
    """
    files = sorted(glob.iglob(os.path.join(in_directory, "*.csv")))

    print("collecting ENT CPT codes...")
    ent_cpt_codes = set()
    for file_path in files:
        df = pd.read_csv(file_path, low_memory=False, usecols=['CPT', 'SURGSPEC'])
        df['CPT'] = standardize_cpt(df['CPT'])
        df['SURGSPEC'] = df['SURGSPEC'].astype(str)
        year_codes = set(
            df[df['SURGSPEC'] == "Otolaryngology (ENT)"]['CPT'].dropna().unique()
        )
        ent_cpt_codes |= year_codes
        print(f"  {Path(file_path).stem}: {len(year_codes)} ENT codes, {len(ent_cpt_codes)} total so far")

    frames = []
    for file_path in files:
        df = pd.read_csv(file_path, low_memory=False, usecols=['CPT', 'SURGSPEC', 'OTHERCPT1'
                                                               , 'OTHERCPT2', 'OTHERCPT3' , 'OTHERCPT4',
                                                               'OTHERCPT5', 'OTHERCPT6', 'OTHERCPT7', 'OTHERCPT8',
                                                               'OTHERCPT9', 'OTHERCPT10'])
        df['CPT'] = standardize_cpt(df['CPT'])
        df = df[df['CPT'].isin(ent_cpt_codes)]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    cpt_counts = find_ent_cpt_counts(combined).sort_values(by='exact_ENT_count', ascending=False)
    cpt_counts.to_csv(cpt_out_path, index=False)
    
    return cpt_counts

def compare_cpt_files(file1path: str, file2path: str, outpath: str):
    """
    compares the 930 CPT codes attributed to ENT procedures with given list from Dr. Sina Torabi
    """
    my_cpt = pd.read_csv(file1path).sort_values(by='exact_ENT_count', ascending=False)
    sina_cpt = pd.read_excel(file2path)['CPT Code']

    sina_set = set(sina_cpt.astype(str))
    my_cpt['AGREE'] = my_cpt['CPT'].astype(str).isin(sina_set).map({True: "TRUE", False: "FALSE"})

    my_cpt.to_csv(outpath, index=False)


def get_final_cpts(file1path: str, file2path: str, file0506path: str, file22path: str, output_path: str) -> set:
    """
    compares the 930 CPT codes attributed to ENT procedures with given list from Dr. Sina Torabi and retrieves
    common codes with >100 solely ENT cases
    """
    
    comparison_cpt = pd.read_csv(file1path)
    sina_cpt = pd.read_excel(file2path)
    rvu0506 = pd.read_csv(file0506path, usecols=['CPT', 'WORKRVU'])
    rvu22 = pd.read_csv(file22path)

    sina_cpt['CPT Code'] = standardize_cpt(sina_cpt['CPT Code'])
    comparison_cpt['CPT'] = standardize_cpt(comparison_cpt['CPT'])
    rvu0506['CPT'] = standardize_cpt(rvu0506['CPT'])
    rvu22['CPT'] = standardize_cpt(rvu22['CPT'])

    comparison_cpt = comparison_cpt[(comparison_cpt['ent_alone_count'] > 100) & (comparison_cpt['AGREE'] == True)]

    rvu_0506_lookup = (
        rvu0506[['CPT', 'WORKRVU']]
        .dropna(subset=['CPT', 'WORKRVU'])
        .drop_duplicates(subset=['CPT'])
        .rename(columns={'WORKRVU': 'work_rvu_2005_2006'})
    )

    rvu_2022_lookup = (
        rvu22[['CPT', 'WORKRVU']]
        .dropna(subset=['CPT', 'WORKRVU'])
        .drop_duplicates(subset=['CPT'])
        .rename(columns={'WORKRVU': 'work_rvu_2022'})
    )

    final = sina_cpt[sina_cpt['CPT Code'].isin(comparison_cpt['CPT'])]
    final = pd.merge(
        comparison_cpt[['CPT', 'count']],
        sina_cpt.rename(columns={'CPT Code':'CPT'}),
        on='CPT',
        how='inner'
    )

    final = pd.merge(
        final,
        rvu_0506_lookup,
        on='CPT',
        how='left'
    )

    final = pd.merge(
        final,
        rvu_2022_lookup,
        on='CPT',
        how='left'
    )

    final.to_csv("data/final_CPT_1.csv", index=False)
    print(f"Saved {final.shape[0]} CPT codes to {output_path}")

    return set(final['CPT'])

def find_cpts(in_directory: str, cpts: list):
    files = sorted(glob.iglob(os.path.join(in_directory, "*.csv")))
    frames = []
    print("step 1: concatenating")
    cols = ['CPT'] + [f'OTHERCPT{i}' for i in range(1, 11)]

    for file_path in files:
        df = pd.read_csv(file_path, low_memory=False, usecols=cols)
        df['CPT'] = standardize_cpt(df['CPT'])
        for col in cols:
            df[col] = standardize_cpt(df[col])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    print("step 2: looking")
    
    result_rows = []

    for cpt in cpts:
        print(f"looking for {cpt}")

        row = {'CPT': cpt}
        found_primary = (combined['CPT'] == cpt).any()
        row['found_in_CPT'] = found_primary

        for i in range(1, 11):
            col = f'OTHERCPT{i}'
            row[f'found_in_{col}'] = (combined[col] == cpt).any()

        if not found_primary:
            mask = False
            for i in range(1, 11):
                mask |= (combined[f'OTHERCPT{i}'] == cpt)

            subset = combined[mask]
            top3 = (
                subset['CPT']
                .value_counts()
                .head(3)
                .index
                .tolist()
            )

            for idx in range(3):
                row[f'top_primary_{idx+1}'] = top3[idx] if idx < len(top3) else None
        else:
            row['top_primary_1'] = None
            row['top_primary_2'] = None
            row['top_primary_3'] = None

        result_rows.append(row)

    found_df = pd.DataFrame(result_rows)

    found_df.to_csv("cpt_lookup_results.csv", index=False)
    return found_df
        
def combine_hcup_nsqip(file1path: str, file2path: str, file0506path: str, file22path: str,
                   hcuppath1: str, hcuppath2: str, final29path: str, 
                   nsqip_dir: str, output_path: str) -> set:

    sina_cpt = pd.read_excel(file2path)
    rvu0506 = pd.read_csv(file0506path, usecols=['CPT', 'WORKRVU'])
    rvu22 = pd.read_csv(file22path, usecols=['CPT', 'WORKRVU'])
    hcup_codelist = pd.read_csv(hcuppath1)  # 32 HCUP codes + metadata
    hcup_counts   = pd.read_csv(hcuppath2)  # HCUP counts for the 29 NSQIP codes
    final29 = pd.read_csv(final29path)

    for df, col in [
        (sina_cpt, 'CPT Code'), (rvu0506, 'CPT'), (rvu22, 'CPT'),
        (hcup_codelist, 'CPT'), (hcup_counts, 'CPT'), (final29, 'CPT')
    ]:
        df[col] = standardize_cpt(df[col])

    # ── compute NSQIP_COUNT from yearly raw files ─────────────
    print("counting NSQIP occurrences across yearly files...")
    files = sorted(glob.iglob(os.path.join(nsqip_dir, "*.csv")))
    nsqip_full = pd.concat([
        pd.read_csv(f, low_memory=False, usecols=['CPT'])
        for f in files
    ], ignore_index=True)
    nsqip_full['CPT'] = standardize_cpt(nsqip_full['CPT'])
    nsqip_counts = (
        nsqip_full['CPT'].value_counts()
        .reset_index()
    )
    nsqip_counts.columns = ['CPT', 'NSQIP_COUNT']

    # ── build union of codes ──────────────────────────────────
    nsqip_cpts = final29[['CPT']].copy()
    nsqip_cpts['IN_FINAL_NSQIP'] = True

    hcup_cpts = hcup_codelist[['CPT']].copy()
    hcup_cpts['IN_FINAL_HCUP'] = True

    all_codes = pd.merge(nsqip_cpts, hcup_cpts, on='CPT', how='outer')
    all_codes['IN_FINAL_NSQIP'] = all_codes['IN_FINAL_NSQIP'].fillna(False)
    all_codes['IN_FINAL_HCUP']  = all_codes['IN_FINAL_HCUP'].fillna(False)
    print(f"Total unique codes after union: {len(all_codes)}")

    # ── metadata: hcup_codelist first, fill gaps from sina ────
    hcup_meta_cols = [c for c in ['CPT', 'Long Desc', 'Intra Time', 'Most Recent RUC Review']
                      if c in hcup_codelist.columns]
    all_codes = all_codes.merge(hcup_codelist[hcup_meta_cols], on='CPT', how='left')

    sina_subset = (
        sina_cpt[['CPT Code', 'Long Desc', 'Intra Time', 'Most Recent RUC Review']]
        .rename(columns={'CPT Code': 'CPT'})
    )
    all_codes = all_codes.merge(sina_subset, on='CPT', how='left', suffixes=('', '_sina'))
    for col in ['Long Desc', 'Intra Time', 'Most Recent RUC Review']:
        sina_col = f'{col}_sina'
        if sina_col in all_codes.columns:
            all_codes[col] = all_codes[col].fillna(all_codes[sina_col])
            all_codes.drop(columns=[sina_col], inplace=True)

    # ── NSQIP_COUNT ───────────────────────────────────────────
    all_codes = all_codes.merge(nsqip_counts, on='CPT', how='left')
    all_codes['NSQIP_COUNT'] = all_codes['NSQIP_COUNT'].fillna(0).astype(int)

    # ── HCUP_COUNT: hcup_counts for NSQIP codes,
    #               hcup_codelist for HCUP-only codes ──────────
    hcup_count_combined = pd.concat([
        hcup_counts[['CPT', 'HCUP_COUNT']],       # 29 NSQIP codes — takes priority
        hcup_codelist[['CPT', 'HCUP_COUNT']]       # 32 HCUP codes — fills the rest
    ]).drop_duplicates(subset='CPT', keep='first')

    all_codes = all_codes.merge(hcup_count_combined, on='CPT', how='left')
    all_codes['HCUP_COUNT'] = all_codes['HCUP_COUNT'].fillna(0).astype(int)

    # ── RVU lookups ───────────────────────────────────────────
    rvu_0506_lookup = (
        rvu0506.dropna(subset=['CPT', 'WORKRVU'])
        .drop_duplicates(subset=['CPT'])
        .rename(columns={'WORKRVU': 'work_rvu_2005_2006'})
    )
    rvu_2022_lookup = (
        rvu22.dropna(subset=['CPT', 'WORKRVU'])
        .drop_duplicates(subset=['CPT'])
        .rename(columns={'WORKRVU': 'work_rvu_2022'})
    )
    all_codes = all_codes.merge(rvu_0506_lookup, on='CPT', how='left')
    all_codes = all_codes.merge(rvu_2022_lookup, on='CPT', how='left')

    all_codes.to_csv(output_path, index=False)
    print(f"Saved {all_codes.shape[0]} CPT codes to {output_path}")
    print(f"In NSQIP final: {all_codes['IN_FINAL_NSQIP'].sum()}")
    print(f"In HCUP final:  {all_codes['IN_FINAL_HCUP'].sum()}")
    print(f"In both:        {(all_codes['IN_FINAL_NSQIP'] & all_codes['IN_FINAL_HCUP']).sum()}")

    return set(all_codes['CPT'])

if __name__ == "__main__":
    #get_cpts("data/nsqip_new",  "data/nsqip/ent_cpt_codes.csv")
    #compare_cpt_files("data/nsqip/ent_cpt_codes.csv", "data/sina_ENT.xlsx", "data/CPT_comparison.csv")    
    #get_final_cpts("data/CPT_comparison.csv", "data/sina_ENT.xlsx", "data/nsqip/2006.csv", "data/nsqip/2022.csv", "data/final_CPT_1.csv")
    #df = pd.read_excel("C:/Users/melod/Desktop/prog/170a/proj/ent-capstone/data/sina_ENT.xlsx")
    #find_cpts("data/nsqip_new", list(df["CPT Code"]))
    """
    combine_hcup_nsqip(
        file1path="data/CPT_comparison.csv",
        file2path="data/sina_ENT.xlsx",
        file0506path="data/nsqip/2006.csv",
        file22path="data/nsqip/2022.csv",
        hcuppath1="data/table 1 HCUP.csv",
        hcuppath2="data/hcup_counts.csv",
        final29path="C:/Users/melod/Desktop/prog/170a/proj/ent-capstone/data/final_CPT_1.csv",
        nsqip_dir="data/nsqip_new",
        output_path="data/final_CPT_FULL.csv"
    )
    """