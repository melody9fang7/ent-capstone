import pandas as pd
from filtering import standardize_cpt, load_cpt_list

def drop_columns():
    """
    Returns a list of columns to drop from merged dataset.
    """
    drop_cols = []
    drop_cols += [f"PR{i}" for i in range(1, 16)]
    drop_cols += [f"PRCCS{i}" for i in range(1, 16)]
    drop_cols += [f"PRDAY{i}" for i in range(1, 16)]
    drop_cols += ["NPR"]

    return drop_cols

def create_volume_table(filtered_file, total_cases_file, cpt_list, output_file):
    """
    Creates yearly CPT volume table.
    PRIMARY_COUNT / TOTAL_HCUP_CASES
    """

    df = pd.read_csv(filtered_file, low_memory=False)
    totals_df = pd.read_csv(total_cases_file)

    df["CPT1"] = standardize_cpt(df["CPT1"])
    df["AYEAR"] = pd.to_numeric(df["AYEAR"], errors="coerce").astype(int)

    totals_df["AYEAR"] = pd.to_numeric(totals_df["AYEAR"], errors="coerce").astype(int)
    cpt_codes = load_cpt_list(cpt_list)
    rows = []

    print("Creating yearly volume table...")

    years = sorted(df["AYEAR"].dropna().unique())
    unique_cpts = sorted(set(df["CPT1"].dropna().unique()) & cpt_codes)

    for year in years:
        print(f"Processing {year}...")

        year_df = df[df["AYEAR"] == year]
        total_cases = totals_df.loc[totals_df["AYEAR"] == year, "TOTAL_CASES"].iloc[0]

        for cpt in unique_cpts:
            primary_count = (year_df["CPT1"] == cpt).sum()
            normalized_volume = (primary_count / total_cases) * 100

            rows.append({"CPT": cpt, "AYEAR": year, "PRIMARY_COUNT": primary_count,"TOTAL_HCUP_CASES": total_cases,
                        "NORMALIZED_VOLUME": normalized_volume}) 

    volume_df = pd.DataFrame(rows)
    volume_df = volume_df.sort_values(["CPT", "AYEAR"])
    volume_df.to_csv(output_file, index=False)

    print("\nSaved yearly volume table.")

    return volume_df
 
def filter_volume_chunk(chunk, cpt_codes, drop_cols):
     """
     Filters a chunk of the merged dataset, only keeping rows with CPT codes of interest and 
     dropping any unnecessary columns. Also filters out rows before 2008.
     """
     chunk["AYEAR"] = pd.to_numeric(chunk["AYEAR"], errors = "coerce")
     chunk = chunk[chunk["AYEAR"] >= 2008]
     chunk = chunk.dropna(subset = ["AYEAR"])

     cpt_cols = [f"CPT{i}" for i in range(1, 51)]

     # standardize all cpts
     for col in cpt_cols:
        chunk[col] = standardize_cpt(chunk[col])

     # checks if row has any cpt code
     mask = pd.Series(False, index = chunk.index)

     for col in cpt_cols:
        mask |= chunk[col].isin(cpt_codes)

     filtered_chunk = chunk[mask]
     filtered_chunk = filtered_chunk.drop(columns = drop_cols, errors = "ignore")

     return filtered_chunk

def filter_hcup_volume(hcup_merged, cpt_list, output_file, chunk_size = 150000):
    """
    Counts total cases for each year.
    """

    cpt_codes = load_cpt_list(cpt_list)
    drop_cols = drop_columns()

    first_chunk = True
    kept_rows = 0

    yearly_count = {}

    print("Starting volume filtering...")

    for chunk in pd.read_csv(hcup_merged, chunksize = chunk_size, low_memory = False):
        # this is for the yearly count for total procedures done (general)
        chunk["AYEAR"] = pd.to_numeric(chunk["AYEAR"], errors = "coerce")
        chunk = chunk.dropna(subset = ["AYEAR"])
        chunk["AYEAR"] = chunk["AYEAR"].astype(int)

        year_counts = (chunk["AYEAR"].value_counts())

        for year, count in year_counts.items():
            if year not in yearly_count:
                yearly_count[year] = 0
            yearly_count[year] += count

        # start actual filtering for ent cases
        filtered_chunk = filter_volume_chunk(chunk, cpt_codes, drop_cols)
        kept_rows += len(filtered_chunk)
        filtered_chunk.to_csv(output_file, mode = "w" if first_chunk else "a", header = first_chunk,index = False)
        first_chunk = False

        print(f"Current rows kept: {kept_rows}")

    print(f"\nFiltering DONE.")
    print(f"Total rows kept: {kept_rows}")

    totals_df = pd.DataFrame({"AYEAR": yearly_count.keys(), "TOTAL_CASES": yearly_count.values()})
    totals_df = totals_df.sort_values("AYEAR")
    totals_df.to_csv("HCUP_total_yearly_cases.csv", index = False)
    print("\nSaved yearly HCUP counts.")

def count_volume_cpts(filtered_file, cpt_list, output_file):
    """
    Counts ENT CPT appearances across all CPT columns.
    """

    df = pd.read_csv(filtered_file, low_memory = False)
    cpt_codes = load_cpt_list(cpt_list)
    cpt_cols = [f"CPT{i}" for i in range(1, 51)]

    counts = {}

    print("Counting CPT appearances...")

    for _, row in df.iterrows():
        # avoids double counting within same section
        primary_seen = set()
        secondary_seen = set()
        other_seen = set()
        total_seen = set()

        cpt1 = str(row["CPT1"]).strip()
        if cpt1 in cpt_codes:
            primary_seen.add(cpt1)
            total_seen.add(cpt1)

        cpt2 = str(row["CPT2"]).strip()
        if cpt2 in cpt_codes:
            secondary_seen.add(cpt2)
            total_seen.add(cpt2)

        for col in cpt_cols[2:]:
            cpt = str(row[col]).strip()
            if cpt in cpt_codes:
                other_seen.add(cpt)
                total_seen.add(cpt)

        all_cpts = (primary_seen | secondary_seen | other_seen)

        for cpt in all_cpts:
            if cpt not in counts:
                counts[cpt] = {"PRIMARY_COUNT": 0, "SECONDARY_COUNT": 0, "OTHER_COUNT": 0, "TOTAL_COUNT": 0}

        for cpt in primary_seen:
            counts[cpt]["PRIMARY_COUNT"] += 1

        for cpt in secondary_seen:
            counts[cpt]["SECONDARY_COUNT"] += 1

        for cpt in other_seen:
            counts[cpt]["OTHER_COUNT"] += 1

        for cpt in total_seen:
            counts[cpt]["TOTAL_COUNT"] += 1

    rows = []

    for cpt, vals in counts.items():
        rows.append({
            "CPT": cpt,
            "PRIMARY_COUNT":
                vals["PRIMARY_COUNT"],

            "SECONDARY_COUNT":
                vals["SECONDARY_COUNT"],

            "OTHER_COUNT":
                vals["OTHER_COUNT"],

            "TOTAL_COUNT":
                vals["TOTAL_COUNT"]
        })

    counts_df = pd.DataFrame(rows)

    counts_df = counts_df.sort_values("TOTAL_COUNT", ascending = False)

    counts_df.to_csv(output_file, index = False)

    print("Done.")
    print(counts_df.head())

    return counts_df

def main():
    hcup_merged = "HCUP_merged.csv" # output from merge.py
    cpt_list = "ENT Codes since 1997.xlsx" # excel file with RUC data
    output_file = "HCUP_volume_filtered.csv" 
    total_cases = "HCUP_total_yearly_cases.csv"
    volume_c = "hcup_volume_counts.csv"
    volume_t = "hcup_volume_table.csv"
    filter_hcup_volume(hcup_merged, cpt_list, output_file)
    count_volume_cpts(output_file, cpt_list, volume_c)
    create_volume_table(output_file, total_cases, cpt_list, volume_t)

if __name__ == "__main__":
    main()
