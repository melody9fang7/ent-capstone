"""
Script to filter the merged HCUP dataset and prepare it for
operative time analysis.
"""
import pandas as pd

# Functions for filtering the merged HCUP dataset. 

def standardize_cpt(series: pd.Series) -> pd.Series:
    """
    Forces formatting of CPT codes.
    """
    def _to_str(x):
        if pd.isna(x):
            return None
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return str(x).strip()
    return series.apply(_to_str)

def load_cpt_list(file: str) -> set:
    """
    Loads CPT codes from given excel file and returns a unique
    set of standardized CPT codes.
    """
    df = pd.read_excel(file)
    df["CPT Code"] = standardize_cpt(df["CPT Code"])

    codes = set(df["CPT Code"])
    #codes.discard("nan")

    return codes

def drop_columns():
    """
    Returns a list of columns to drop from merged dataset.
    """
    drop_cols = []

    drop_cols += [f"CPT{i}" for i in range(3, 51)]
    drop_cols += [f"CPTCCS{i}" for i in range(3, 51)]
    drop_cols += [f"CPTDAY{i}" for i in range (3, 51)]
    drop_cols += [f"PR{i}" for i in range(1, 16)]
    drop_cols += [f"PRCCS{i}" for i in range(1, 16)]
    drop_cols += [f"PRDAY{i}" for i in range(1, 16)]
    drop_cols += ["NPR"]

    return drop_cols

def filter_chunk(chunk: pd.DataFrame, cpt_codes: set, drop_cols: list, require_valid_ortime : bool = False) -> pd.DataFrame:
    """
    Filters a chunk of merged dataset to only include solo cases within the
    valid year range, dropping any unnecessary columns.
    """
    chunk["AYEAR"] = pd.to_numeric(chunk["AYEAR"], errors="coerce")
    chunk["NCPT"] = pd.to_numeric(chunk["NCPT"], errors="coerce")

    # only keep solo procedures with AYEAR >= 2008
    chunk = chunk[chunk["AYEAR"] >= 2008]
    chunk = chunk[chunk["NCPT"] == 1]

    # if you want to include invalid (missing or <= 0) ortime or not
    if require_valid_ortime:
        chunk["ORTIME"] = pd.to_numeric(chunk["ORTIME"], errors="coerce")
        chunk = chunk.dropna(subset=["ORTIME"])
        chunk = chunk[chunk["ORTIME"] > 0]

    chunk["CPT1"] = standardize_cpt(chunk["CPT1"])

    # keep rows where CPT1 (primary procedure) is in the ent cpt code list
    chunk = chunk[chunk["CPT1"].isin(cpt_codes)]

    chunk = chunk.drop(columns=drop_cols)

    return chunk

def filter_hcup(hcup_merged, cpt_list, output_file, require_valid_ortime = False, chunk_size = 150000):
    """
    Filters the merged HCUP dataset.
    - Only keeps solo cases (NCPT = 1) with AYEAR >= 2008. 
    - Keeps rows where CPT1 (primary procedure) is in the ENT CPT code list. 
    - Drops any unnecessary columns.
    """
    cpt_codes = load_cpt_list(cpt_list)
    drop_cols = drop_columns()
    
    first_chunk = True
    kept_rows = 0

    print("Starting filtering...")
    for chunk in pd.read_csv(hcup_merged, chunksize = chunk_size, low_memory = False):
        filtered_chunk = filter_chunk(chunk, cpt_codes, drop_cols, require_valid_ortime = require_valid_ortime)

        kept_rows += len(filtered_chunk)

        filtered_chunk.to_csv(output_file, mode = "w" if first_chunk else "a", header = first_chunk,
                              index = False)
        
        first_chunk = False

    print(f"\nFiltering DONE.")
    print(f"Total rows kept: {kept_rows}")

def save_cpt_counts(input_file, output_file):
    """
    Creates CPT1 count table from filtered HCUP file.
    """

    df = pd.read_csv(input_file)

    # standardize CPT1
    df["CPT1"] = standardize_cpt(df["CPT1"])

    # count CPT1 values
    count_df = (df["CPT1"].value_counts().reset_index())

    count_df.columns = ["CPT1", "Count"]

    count_df.to_csv(output_file, index=False)

    print("Unique CPTs:", count_df["CPT1"].nunique())
    print(f"Saved counts to {output_file}")
    return count_df


# Filter the ENT CPT excel file. This is created a filtered version that will be used for
# later analysis. This filtering is done after HCUP filtering to keep only CPT codes
# that can actually be used for analysis.

def filter_ent_codes(excel_file, cpt_list_file, output_csv, min_count = 100):
    """
    Filters ENT CPT excel file using CPT codes from cpt_counts.csv where Count > min_count. Creates a filtered
    version with valid CPT codes.
    """
    df = pd.read_excel(excel_file)
    cpt_df = pd.read_csv(cpt_list_file)

    cpt_df = cpt_df[cpt_df["Count"] > min_count]
    keep_cpts = set(cpt_df["CPT1"].fillna("").astype(str).str.replace(".0", "", regex=False).str.strip())
    df["CPT Code"] = (df["CPT Code"].fillna("").astype(str).str.replace(".0", "", regex=False).str.strip())

    filtered_df = df[df["CPT Code"].isin(keep_cpts)]
    filtered_df.to_csv(output_csv, index=False)

    print("Original rows:", len(df))
    print("Filtered rows:", len(filtered_df))
    print("Done")

    return filtered_df

# Additional functions for table creation and wRVU extraction from NSQIP dataset.
# Not used for filtering HCUP dataset, optional to run.

# table creation (figure 1)
def create_hcup_table(hcup_file, sina_file, output_file):
    """
    Creates table detailing HCUP stats.
    """

    hcup = pd.read_csv(hcup_file, low_memory=False)
    hcup["CPT1"] = standardize_cpt(hcup["CPT1"])

    # HCUP counts
    hcup_counts = (hcup["CPT1"].value_counts().rename_axis("CPT").reset_index(name = "HCUP_count"))

    ruc = pd.read_csv(sina_file)
    ruc["CPT Code"] = standardize_cpt(ruc["CPT Code"])
    ruc = ruc[["CPT Code", "Long Desc", "Intra Time", "Most Recent RUC Review"]]
    ruc = ruc.rename(columns={"CPT Code": "CPT"})

   
    table = pd.merge(ruc, hcup_counts, on="CPT", how="inner")

    table["NSQIP_count"] = pd.NA
    table["work_rvu_2005_2006"] = pd.NA
    table["work_rvu_2022"] = pd.NA

    table = table[
        ["CPT", "NSQIP_count", "HCUP_count", "Long Desc", "Intra Time", 
        "Most Recent RUC Review","work_rvu_2005_2006","work_rvu_2022",]
    ]

    
    table = table.sort_values("CPT")
    table.to_csv(output_file, index=False)

    print("Table 1 HCUP file created")
    print(f"Rows: {len(table)}")


def extract_wrvu(table_file, nsqip_file, output_file, chunk_size = 150000):
    """
    Extracts 2006 and 2022 wRVUs from NSQIP dataset for CPT codes.
    """

    table = pd.read_csv(table_file, low_memory = False)
    table["CPT"] = standardize_cpt(table["CPT"])
    keep_cpts = set(table["CPT"])

    wrvu_2006 = {}
    wrvu_2022 = {}

    print("Starting wRVU extraction...")
    for chunk in pd.read_csv(nsqip_file, chunksize = chunk_size, low_memory = False):
        chunk["CPT"] = standardize_cpt(chunk["CPT"])
        chunk["PUFYEAR"] = pd.to_numeric(chunk["PUFYEAR"], errors = "coerce").astype("Int64")
        chunk["WORKRVU"] = pd.to_numeric(chunk["WORKRVU"], errors = "coerce")

        for i in range(1, 11):
            chunk[f"OTHERCPT{i}"] = standardize_cpt(chunk[f"OTHERCPT{i}"])
            chunk[f"OTHERWRVU{i}"] = pd.to_numeric(chunk[f"OTHERWRVU{i}"], errors = "coerce")

        main_match = chunk[chunk["CPT"].isin(keep_cpts)]

        for _, row in main_match.iterrows():
            cpt = row["CPT"]
            year = row["PUFYEAR"]
            wrvu = row["WORKRVU"]

            if pd.notna(wrvu):
                if year == 2006 and cpt not in wrvu_2006:
                    wrvu_2006[cpt] = wrvu
                if year == 2022 and cpt not in wrvu_2022:
                    wrvu_2022[cpt] = wrvu

        for i in range(1, 11):

            cpt_col = f"OTHERCPT{i}"
            wrvu_col = f"OTHERWRVU{i}"
            other_match = chunk[chunk[cpt_col].isin(keep_cpts)]

            for _, row in other_match.iterrows():
                cpt = row[cpt_col]
                year = row["PUFYEAR"]
                wrvu = row[wrvu_col]
                if pd.notna(wrvu):
                    if year == 2006 and cpt not in wrvu_2006:
                        wrvu_2006[cpt] = wrvu
                    if year == 2022 and cpt not in wrvu_2022:
                        wrvu_2022[cpt] = wrvu

 
    table["work_rvu_2005_2006"] = (table["CPT"].map(wrvu_2006))
    table["work_rvu_2022"] = (table["CPT"].map(wrvu_2022))

    table.to_csv(output_file, index = False)

    print("\nwRVU extraction DONE.")
    print(f"Rows: {len(table)}")

def search_for_other_counts(hcup_file, output_file):
    """Checks for CPT counts for NSQIP unique CPTs."""

    missing_cpts = ["15731", "21044", "31360", "31365", "31591", "38542", "38700", "38720", "38724", "40810",
                    "40816", "41120", "41130", "41135", "41155","42120", "42420", "42842", "60252", "60254", 
                    "60260", "60270", "60271"]

  
    df = pd.read_csv(hcup_file, low_memory = False)


    df["ORTIME"] = pd.to_numeric(df["ORTIME"], errors="coerce")
    df = df.dropna(subset=["ORTIME"])
    df = df[df["ORTIME"] > 0]

    df["CPT1"] = standardize_cpt(df["CPT1"])

    counts = (df["CPT1"].value_counts().to_dict())
    results = pd.DataFrame({"CPT": missing_cpts})

    results["HCUP_count"] = (results["CPT"].map(counts).fillna(0).astype(int))

    results.to_csv(output_file, index=False)

    print("Done")
    print(results)

    return results

def main():
    # ------------------------
    # HCUP filtering pipeline
    # ------------------------

    # input files
    hcup_merged = "HCUP_merged_extended.csv" # output of merge.py
    ent_codes = "ENT Codes since 1997.xlsx" # excel file with RUC data, with the CPT codes of interest
    
    # output files
    hcup_filtered = "HCUP_filtered_172.csv"
    hcup_counts = "hcup_filtered_172_counts.csv"
    filtered_ent_codes = "filtered_sina2.csv"

    # filter merged HCUP dataset
    filter_hcup(hcup_merged = hcup_merged, cpt_list = ent_codes, output_file = hcup_filtered, require_valid_ortime = True)

    # count CPT codes in filtered HCUP dataset
    save_cpt_counts(input_file = hcup_filtered, output_file = hcup_counts)

    # filter ENT CPT code excel file to get final list of CPT codes for analysis
    filter_ent_codes(excel_file = ent_codes, cpt_list_file = hcup_counts, output_csv = filtered_ent_codes, min_count = 100)

    # ------------------
    # Optional analysis.
    # ------------------

    #table_output = "table1_hcup.csv"
    #nsqip_file = "combined_filtered_930.csv"
    #table_output_with_wrvu = "table1_hcup_wrvu.csv"
    #nsqip_cpt_count = "hcup_nsqip_counts.csv"

    # create table for figure 1
    #create_hcup_table(hcup_file = hcup_filtered, sina_file = filtered_ent_codes, output_file = table_output)

    # extract wRVU values from NSQIP
    #extract_wrvu(table_file = table_output, nsqip_file = nsqip_file, output_file = table_output_with_wrvu)

    # check for counts of NSQIP unique CPTs in HCUP dataset
    #search_for_other_counts(hcup_file = hcup_filtered, output_file = nsqip_cpt_count)

if __name__ == "__main__":
    main()
