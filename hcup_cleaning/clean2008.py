"""
Script to clean and standardize the 2008 HCUP dataset. 

Input: 
A csv file exported from IBM SPSS Statistics. 

To create this file for any of the years, use the load program from 
the following link (if not already in possession): https://hcup-us.ahrq.gov/spssload/spssload_search.jsp and 
click "SASD Current Load Programs". 
Open the file for this year up in IBM SPSS Statistics, and
make any adjustments in order to convert your .asc file into a .sav file. Then, 
export as .csv, and place the file in the same directory as this script.

Output: NY_SASD_2008_CORE_CLEANED.csv
"""

import pandas as pd
import numpy as np
from column_order import COL_ORDER

input_file = "2008.csv"
output_file = "NY_SASD_2008_CORE_CLEANED.csv"
chunksize = 100_000   
print("Running...")

# -----------------------------------------------------
# Columns available (and unavailable) for 2008 dataset.
# -----------------------------------------------------
existing_cols = ["AGE", "AGEDAY", "AGEMONTH", "AHOUR", "AYEAR", "DHOUR", "DSHOSPID", "FEMALE",
                 "DX_Visit_Reason1", "KEY", "HISPANIC_X", "HCUP_OS",
                 "LOS", "LOS_X", "NCPT", "PAY1", "PAY1_X", "ORTIME", "RACE", "TOTCHG", "NDX", "ANESTH"]

existing_cols.extend(
    [f"CPT{i}" for i in range(1, 21)] +
    [f"CPTCCS{i}" for i in range(1, 21)] +
    [f"DX{i}" for i in range(1, 16)] +
    [f"CPTM1_{i}" for i in range(1, 21)] +
    [f"CPTM2_{i}" for i in range(1, 21)]
)

nonexisting_cols = ["DURATION",
                    "HCUP_SURGERY_BROAD_CPT", "HCUP_SURGERY_NARROW_CPT", "HISPANIC", "OS_TIME",
                    "PAY2", "PAY2_X", "PAY3", "PAY3_X", "I10_DX_VISIT_Reason", "NPR"]

nonexisting_cols.extend(
    [f"CPT{i}" for i in range(21, 51)] +
    [f"CPTCCS{i}" for i in range(21, 51)] +
    [f"CPTM1_{i}" for i in range(21, 41)] +
    [f"CPTM2_{i}" for i in range(21, 31)] +
    [f"CPTDAY{i}" for i in range(1, 51)] +
    [f"PR{i}" for i in range(1, 16)] +
    [f"PRCCS{i}" for i in range(1, 16)] +
    [f"PRDAY{i}" for i in range(1, 16)] +
    [f"I10_DX{i}" for i in range(1, 16)] 
)

print(len(set(existing_cols + nonexisting_cols)))

# -----------------------------------------------
# Convert columns to the appropriate data types
# and rename columns as necessary.
# -----------------------------------------------

cols_to_numeric = [
    "AGE", "AGEDAY", "AGEMONTH", "AHOUR", "AYEAR",
    "DHOUR", "FEMALE", "LOS", "LOS_X",
    "ORTIME", "PAY1", "PAY2", "PAY3",
    "RACE", "HISPANIC", "HISPANIC_X",
    "TOTCHG", "HCUP_OS", "DURATION",
    "HCUP_SURGERY_BROAD_CPT", "HCUP_SURGERY_NARROW_CPT",
    "OS_TIME", "NCPT", "NPR", "NDX", "ANESTH"
]

cols_to_numeric.extend(
    [f"CPTDAY{i}" for i in range(1, 51)] +
    [f"PRDAY{i}" for i in range(1, 16)]
)

cols_to_string = [
    "DSHOSPID", "KEY",
    "PAY1_X", "PAY2_X", "PAY3_X",
    "DX_VISIT_Reason1", "I10_DX_VISIT_Reason"
]

cols_to_string.extend(
    [f"PRCCS{i}" for i in range(1, 16)] +
    [f"PR{i}" for i in range(1, 16)] +
    [f"CPTCCS{i}" for i in range(1, 51)] +
    [f"CPTM1_{i}" for i in range(1, 41)] +
    [f"CPTM2_{i}" for i in range(1, 31)] +
    [f"CPT{i}" for i in range(1, 51)] +
    [f"DX{i}" for i in range(1, 16)] +
    [f"I10_DX{i}" for i in range(1, 16)]
)

rename_map = {
    "DX1": "I9_DX1",
    "DX2": "I9_DX2",
    "DX3": "I9_DX3",
    "DX4": "I9_DX4",
    "DX5": "I9_DX5",
    "DX6": "I9_DX6",
    "DX7": "I9_DX7",
    "DX8": "I9_DX8",
    "DX9": "I9_DX9",
    "DX10": "I9_DX10",
    "DX11": "I9_DX11",
    "DX12": "I9_DX12",
    "DX13": "I9_DX13",
    "DX14": "I9_DX14",
    "DX15": "I9_DX15",
    "DX_Visit_Reason1": "I9_DX_VISIT_Reason",
    "I10_NDX": "NDX"
}

exclude_cols = [f"CPTDAY{i}" for i in range(1, 51)] + ["LOS_X"]

# -------------------------------------------------------------
# Read the .csv file in chunks, clean each chunk, 
# and write to output file. Change input_file and output_file
# variables at the top of this script to match your file names.
# -------------------------------------------------------------

first_chunk = True
total_rows = 0

for i, chunk in enumerate(
    pd.read_csv(
        input_file,
        usecols=existing_cols,
        na_values=[" ", "", ".", ".A", ".B", ".C", "i", "invl", "incn", "incn2"],
        chunksize=chunksize,
        low_memory=False
    )
):
    print(f"Cleaning chunk {i + 1}...")

    # add columns missing from this year
    for col in nonexisting_cols:
        chunk[col] = np.nan

    for col in cols_to_numeric:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    for col in cols_to_string:
        if col in chunk.columns:
            chunk[col] = chunk[col].astype("string").str.strip()

    chunk = chunk.rename(columns=rename_map)

    chunk = chunk[chunk["KEY"].notna()]

    # remove negative values except CPTDAY columns and LOS_X
    cols_to_fix = [
        col for col in cols_to_numeric
        if col not in exclude_cols and col in chunk.columns
    ]
    chunk[cols_to_fix] = chunk[cols_to_fix].mask(chunk[cols_to_fix] < 0)

    chunk["AYEAR"] = 2008

    chunk = chunk[COL_ORDER]

    chunk.to_csv(
        output_file,
        mode="w" if first_chunk else "a",
        index=False,
        header=first_chunk
    )

    total_rows += len(chunk)
    first_chunk = False

print("Done!")
print(f"Saved to: {output_file}")
print(f"Total cleaned rows written: {total_rows}")
