import pandas as pd
import matplotlib as plt
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

    # avg # of procedures where primary procedure was ENT, but >1 procedure was performed
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
 
def get_cpts(in_directory: str, cpt_out_path: str) -> set:
    """
    given the path to the directory of CSV files for each year, creates filtered CSV for each year and combined filterd CSV for
    all years in the given output directory. returns set of CPT codes.
    """
    files = sorted(glob.iglob(os.path.join(in_directory, "*.csv")))

    print("Pass 1: collecting ENT CPT codes...")
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
    
    return set(cpt_counts['CPT'])

def compare_cpt_files(file1path: str, file2path: str):
    """
    compares the 930 CPT codes attributed to ENT procedures with given list from Dr. Sina Torabi
    """
    my_cpt = pd.read_csv(file1path).sort_values(by='exact_ENT_count', ascending=False)
    sina_cpt = pd.read_excel(file2path)['CPT Code']

    sina_set = set(sina_cpt.astype(str))
    my_cpt['AGREE'] = my_cpt['CPT'].astype(str).isin(sina_set).map({True: "TRUE", False: "FALSE"})

    my_cpt.to_csv("data/CPT_comparison.csv", index=False)


def get_top10_cpts(df: pd.DataFrame) -> list:
    """
    returns a list of the top 10 CPT codes by volume
    """
    return df['CPT'].value_counts().head(10).index.tolist()


def plot_top10_cpt_stacked(df, top_10_cpts) -> None:
    """
    given a list of the top 10 cpt codes, creates a stacked area chart of the procedural volume each code over time
    """
    top_10_cpts = get_top10_cpts(df)
    df_top = df[df['CPT'].isin(top_10_cpts)]
    pivot = df_top.groupby(['PUFYEAR', 'CPT']).size().unstack(fill_value=0)
    pivot = pivot[top_10_cpts] 

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind='area', ax=ax, alpha=0.7, stacked=True, colormap='tab10')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Cases', fontsize=12)
    ax.set_title('Top 10 ENT Procedures Over Time (Stacked Area)', fontsize=14, fontweight='bold')
    ax.legend(title='CPT Code', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figs/top10_cpt_stacked.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Top 10 CPTs (by total volume): {', '.join(map(str, top_10_cpts))}")

if __name__ == "__main__":
    get_cpts("data/nsqip_new",  "data/nsqip/ent_cpt_codes.csv")
    compare_cpt_files("data/nsqip/ent_cpt_codes.csv", "data/sina_ENT.xlsx")
    
    df = pd.read_csv("data/nsqip/ent_cpt_codes.csv")

    counts = df['count']
    mean_count = counts.mean()
    median_count = counts.median()
    num_below_10 = df[df['count'] < 10].shape[0]

    print(f"total: {df.shape[0]}")
    print(f"Mean count: {mean_count:.2f}")
    print(f"Median count: {median_count}")
    print(f"Number of CPT codes with count < 10: {num_below_10}")