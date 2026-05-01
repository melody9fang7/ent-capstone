import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
from cpt_codes import get_cpts, get_final_cpts

_RACE_KEYWORDS = {
    'White':                              'White',
    'Black or African American':          'Black',
    'Black':                              'Black',
    'Asian':                              'Asian',
    'American Indian':                    'AIAN',
    'Alaska Native':                      'AIAN',
    'Native Hawaiian':                    'NHPI',
    'Pacific Islander':                   'NHPI',
    'Some Other Race':                    'Other',
}

_RACE_NEW_TRUNCATION_MAP = {
    'American Indian or Alaska':                          'American Indian or Alaska Native',
    'Native Hawaiian or Pacifi':                          'Native Hawaiian or Other Pacific Islander',
    'Black or African American,American Indian or Alask': 'Black or African American,American Indian or Alaska Native',
    'White,Native Hawaiian or Other Pacific Islander,As': 'White,Native Hawaiian or Other Pacific Islander,Asian',
}

def categorize_race(val: str) -> str:
    """
    Given a raw race string (already stripped of Hispanic info),
    return a broad category. Comma = multiracial.
    """
    if pd.isna(val) or str(val).strip().lower() in ('', 'nan', 'unknown', 'unknown/not reported'):
        return 'Unknown'
    val = str(val).strip()
    if ',' in val:
        return 'Multiracial'
    for keyword, category in _RACE_KEYWORDS.items():
        if keyword.lower() in val.lower():
            return category
    return 'Unknown'

def parse_race(val) -> tuple[str, int | None]:
    """
    parses RACE column value into (race_category, hispanic)
    """
    if pd.isna(val):
        return 'Unknown', None
    val = str(val).strip().strip('"').strip("'")
    val_lower = val.lower()

    is_hispanic = 'hispanic' in val_lower

    if is_hispanic:
        parts = val.split(',', 1)
        race_part = parts[1].strip() if len(parts) > 1 else ''
        if not race_part or 'unknown' in race_part.lower() or 'color' in race_part.lower():
            race_part = 'Unknown'
    else:
        race_part = val.replace(', Not of Hispanic Origin', '').strip()

    return categorize_race(race_part), ("Y" if is_hispanic else "N")

def standardize_hispanic(series: pd.Series) -> pd.Series:
    """
    maps yes/no/unknown variants to Y/N/None
    """
    s = series.astype(str).str.strip().str.lower()
    result = [None] * len(s)
    for i, v in enumerate(s):
        if v in {'yes', 'y'}:
            result[i] = 'Y'
        elif v in {'no', 'n'}:
            result[i] = 'N'
    return pd.Series(result, index=series.index)


def standardize_race(df: pd.DataFrame) -> pd.DataFrame:
    """
    make RACE_NEW and ETHNICITY_HISPANIC the only two columns
    """
    df = df.copy()
    has_race     = 'RACE' in df.columns and df['RACE'].notna().any()
    has_race_new = 'RACE_NEW' in df.columns and df['RACE_NEW'].notna().any()

    if has_race and not has_race_new:
        parsed = df['RACE'].apply(parse_race)
        df['RACE_NEW'] = [r for r, _ in parsed]
        df['ETHNICITY_HISPANIC'] = pd.array([h for _, h in parsed])
        df = df.drop(columns=['RACE'])  # ← add this
    elif has_race_new:
        df['RACE_NEW'] = (
            df['RACE_NEW']
            .astype(str).str.strip()
            .replace(_RACE_NEW_TRUNCATION_MAP)
            .apply(categorize_race)
        )
        df['ETHNICITY_HISPANIC'] = standardize_hispanic(df.get('ETHNICITY_HISPANIC', pd.Series([None] * len(df))))
    else:
        print("  WARNING: no RACE or RACE_NEW data found")
        df['RACE_NEW'] = 'Unknown'
        df['ETHNICITY_HISPANIC'] = pd.array([pd.NA] * len(df), dtype="Int8")


    return df

def standardize_cpt(series: pd.Series) -> pd.Series:
    """
    forces formatting of CPT codes
    """
    def _to_str(x):
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return str(x)
 
    return series.apply(_to_str)

def sav_to_csv(in_directory: str, out_directory: str):
    """
    given the path to the directory containing the filtered SAV files for each year, creates and saves CSV for each year in the given
    output directory.
    """
    os.makedirs(out_directory, exist_ok=True)
    sav_files = list(glob.iglob(f"{in_directory}/*.sav"))
 
    if not sav_files:
        raise FileNotFoundError(f"No .sav files found in '{in_directory}'")
 
    for file_path in sav_files:
        stem = Path(file_path).stem
        out_path = os.path.join(out_directory, f"{stem}.csv")
        print(f"  Converting {file_path} → {out_path}")
        pd.read_spss(file_path).to_csv(out_path, index=False)


def build_combined_filtered(in_directory: str, out_directory: str, out_path: str, ent_cpt_codes: set):
    """
    given the path to the directory of CSV files for each year, creates filtered CSV for each year and combined filterd CSV for
    all years in the given output directory.
    """
    os.makedirs(out_directory, exist_ok=True)
    files = sorted(glob.iglob(os.path.join(in_directory, "*.csv")))

    print("/nPass 2: filtering and saving...")
    frames = []
    for file_path in files:
        year = Path(file_path).stem
        df = pd.read_csv(file_path, low_memory=False)
        df.columns = df.columns.str.upper()
        df['PUFYEAR'] = year
        df['CPT'] = standardize_cpt(df['CPT'])
        df = df[df['CPT'].isin(ent_cpt_codes)]
        df = standardize_race(df)
        
        path = os.path.join(out_directory, f"{year}.csv")
        df.to_csv(path, index=False)
        print(f"  {year}.csv → {len(df)} rows kept")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)

    print(f"Combined → {out_path}, shape={combined.shape}")
    return combined

def firstrun(SAV_IN_DIR, CSV_DIR, FILTERED_CSV_DIR, COMBINED_OUT, CPT_OUT) -> pd.DataFrame:
    """
    run either on first run or if you need to reset everything, need SAV files for each year with the correct variables kept.
    """
    #print("/n=== Step 1: SAV → CSV ===")
    #sav_to_csv(SAV_IN_DIR, CSV_DIR)

    #print("/n=== Step 2: Getting CPT codes ===")
    #cpt_codes = get_cpts(CSV_DIR, CPT_OUT)
    #cpt_codes = get_final_cpts("data/CPT_comparison.csv", "data/sina_ent.xlsx", "data/final_CPT_1.csv")

    cpt_codes = pd.read_csv("C:/Users/melod/Desktop/prog/170a/proj/ent-capstone/data/final_CPT_1.csv")['CPT']
    cpt_codes = standardize_cpt(cpt_codes)
    
    print("/n=== Step 3: Build filtered combined CSV ===")
    df = build_combined_filtered(CSV_DIR, FILTERED_CSV_DIR, COMBINED_OUT, cpt_codes)
    return df
 
def main():
    SAV_IN_DIR       = "data/nsqip_filtered_sav"
    CSV_DIR          = "data/nsqip_new" 
    FILTERED_CSV_DIR = "data/nsqip"
    COMBINED_OUT     = "data/nsqip/combined_filtered.csv"
    CPT_OUT          = "data/nsqip/ent_cpt_codes.csv"
    
    df = firstrun(SAV_IN_DIR, CSV_DIR, FILTERED_CSV_DIR, COMBINED_OUT, CPT_OUT)

    #df2 = pd.read_csv(COMBINED_OUT)


if __name__ == "__main__":
    main()