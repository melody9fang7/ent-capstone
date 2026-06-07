"""
Script to clean and standardize the 2016 and 2017 HCUP dataset. Same logic
as 2008.

Input: 
A csv file exported from IBM SPSS Statistics. 

Output:
NY_SASD_2016_CORE_CLEANED.csv and NY_SASD_2017_CORE_CLEANED.csv

Replace the output file with the appropriate year. This file works for both
2016 and 2017 since they have the same structure.
"""

import pandas as pd
import numpy as np
from column_order import COL_ORDER

# used for both 2016 and 2017
input_file = "NY_SASD_2017_CORE.csv"
output_file = "NY_SASD_2017_CORE_CLEANED.csv"
chunksize = 100_000 
print("Running...")

existing_cols = [
    "AGE", "AGEDAY", "AGEMONTH", "AHOUR", "YEAR",
    "DHOUR", "DSHOSPID", "FEMALE",
    "HCUP_SURGERY_BROAD_CPT", "HCUP_SURGERY_NARROW_CPT",
    "HISPANIC_X", "HISPANIC",
    "HCUP_OS", "DURATION",
    "KEY", "LOS", "LOS_X", "NCPT",
    "PAY1", "PAY1_X", "PAY2", "PAY2_X", "PAY3", "PAY3_X",
    "ORTIME", "RACE", "TOTCHG", "OS_TIME",
    "I10_DX_Visit_Reason1", "I10_NDX", "ANESTH"
]

existing_cols.extend(
    [f"CPT{i}" for i in range(1, 51)] +
    [f"CPTDAY{i}" for i in range(1, 51)] +
    [f"CPTCCS{i}" for i in range(1, 51)] +
    [f"I10_DX{i}" for i in range(1, 16)] +
    [f"CPTM1_{i}" for i in range(1, 41)] +
    [f"CPTM2_{i}" for i in range(1, 31)]
)

nonexisting_cols = ["I9_DX_VISIT_Reason", "NPR"]

nonexisting_cols.extend(
    [f"PR{i}" for i in range(1, 16)] +
    [f"PRCCS{i}" for i in range(1, 16)] +
    [f"PRDAY{i}" for i in range(1, 16)] +
    [f"I9_DX{i}" for i in range(1, 16)] 
)

print(len(set(existing_cols + nonexisting_cols)))


cols_to_numeric = [
    "AGE", "AGEDAY", "AGEMONTH", "AHOUR", "YEAR",
    "DHOUR", "FEMALE", "LOS", "LOS_X",
    "ORTIME", "PAY1", "PAY2", "PAY3",
    "RACE", "HISPANIC", "HISPANIC_X",
    "TOTCHG", "HCUP_OS", "DURATION",
    "HCUP_SURGERY_BROAD_CPT", "HCUP_SURGERY_NARROW_CPT",
    "OS_TIME", "NCPT", "NPR", "I10_NDX", "ANESTH"
]

cols_to_numeric.extend(
    [f"CPTDAY{i}" for i in range(1, 51)] +
    [f"PRDAY{i}" for i in range(1, 16)]
)

cols_to_string = [
    "DSHOSPID", "KEY",
    "PAY1_X", "PAY2_X", "PAY3_X",
    "I9_DX_VISIT_Reason", "I10_DX_Visit_Reason1"
]

cols_to_string.extend(
    [f"PRCCS{i}" for i in range(1, 16)] +
    [f"PR{i}" for i in range(1, 16)] +
    [f"CPTCCS{i}" for i in range(1, 51)] +
    [f"CPTM1_{i}" for i in range(1, 41)] +
    [f"CPTM2_{i}" for i in range(1, 31)] +
    [f"CPT{i}" for i in range(1, 51)] +
    [f"I9_DX{i}" for i in range(1, 16)] +
    [f"I10_DX{i}" for i in range(1, 16)]
)

rename_map = {
    "YEAR": "AYEAR",
    "I10_DX_Visit_Reason1": "I10_DX_VISIT_Reason",
    "I10_NDX": "NDX"
}

exclude_cols = [f"CPTDAY{i}" for i in range(1, 101)] + ["LOS_X"]

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

    chunk["AYEAR"] = 2017

    #chunk = chunk[sorted(chunk.columns)]
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
