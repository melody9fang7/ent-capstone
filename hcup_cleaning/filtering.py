import pandas as pd

# clean cpt codes
def standardize_cpt(series: pd.Series) -> pd.Series:
    """
    forces formatting of cpt codes
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

def filter_chunk(chunk: pd.DataFrame, cpt_codes: set, drop_cols: list) -> pd.DataFrame:
    """
    Filters a chunk of merged dataset to only include solo cases within the
    valid year range, dropping any unnecessary columns.
    """
    chunk["AYEAR"] = pd.to_numeric(chunk["AYEAR"], errors="coerce")
    chunk["NCPT"] = pd.to_numeric(chunk["NCPT"], errors="coerce")

    # only keep solo procedures with AYEAR >= 2008
    chunk = chunk[chunk["AYEAR"] >= 2008]
    chunk = chunk[chunk["NCPT"] == 1]

    chunk["CPT1"] = standardize_cpt(chunk["CPT1"])
    #chunk = chunk[chunk["CPT1"] != "nan"] # filter out any "nan" that came from standardization

    # keep rows where CPT1 (primary procedure) is in the ent cpt code list
    chunk = chunk[chunk["CPT1"].isin(cpt_codes)]

    chunk = chunk.drop(columns=drop_cols)

    return chunk

def filter_hcup(hcup_merged, cpt_list, output_file, chunk_size = 150000):
    cpt_codes = load_cpt_list(cpt_list)
    drop_cols = drop_columns()
    
    first_chunk = True
    kept_rows = 0

    print("Starting filtering...")
    for chunk in pd.read_csv(hcup_merged, chunksize = chunk_size, low_memory = False):
        filtered_chunk = filter_chunk(chunk, cpt_codes, drop_cols)

        kept_rows += len(filtered_chunk)

        filtered_chunk.to_csv(output_file, mode = "w" if first_chunk else "a", header = first_chunk,
                              index = False)
        
        first_chunk = False

    print(f"\nFiltering DONE.")
    print(f"Total rows kept: {kept_rows}")

# main filtering from dataset done, from now on is additional filtering

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

def save_cpt_counts(input_file, output_file):
    """
    Creates CPT1 count table from filtered HCUP file.
    """

    df = pd.read_csv(input_file)

    # standardize CPT1
    df["CPT1"] = standardize_cpt(df["CPT1"])

    # remove nan strings
    #df = df[df["CPT1"] != "nan"]

    # count CPT1 values
    count_df = (df["CPT1"].value_counts().reset_index())

    count_df.columns = ["CPT1", "Count"]

    count_df.to_csv(output_file, index=False)

    print("Unique CPTs:", count_df["CPT1"].nunique())
    print(f"Saved counts to {output_file}")
    return count_df

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
    hcup_merged = "HCUP_merged_extended.csv"
    hcup_not_clean = "HCUP_filtered_172.csv"
    nsqip = "combined_filtered_930.csv"
    cpt_list = "ENT Codes since 1997.xlsx"
    output_file = "hcup_nsqip_counts.csv"

    #filter_hcup(hcup_merged, cpt_list, output_file)
    #filter_ent_codes(excel_file="ENT Codes since 1997.xlsx",cpt_list_file="hcup_filtered_172_counts.csv",output_csv="filtered_sina2.csv", min_count=100)
    #create_hcup_table("HCUP_filtered_172_cleaned.csv", "filtered_sina2.csv", "cpt_list.csv")
    #extract_wrvu("cpt_list.csv", "combined_filtered_930.csv", "cpt_list_reformed.csv")
    search_for_other_counts(hcup_not_clean, output_file)

if __name__ == "__main__":
    main()
