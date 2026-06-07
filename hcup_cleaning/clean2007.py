import pandas as pd
import numpy as np
from column_order import COL_ORDER

"""
Script to clean and standardize the 2007 HCUP dataset. 
Only year that uses ICD-9 codes, so columns are renamed to match the 2008+ datasets.

Input: 
A csv file exported from IBM SPSS Statistics. 

To create this file for any of the years, use the load program from 
the following link (if not already in possession): https://hcup-us.ahrq.gov/spssload/spssload_search.jsp and 
click "SASD Current Load Programs". 
Open the file for this year up in IBM SPSS Statistics, and
make any adjustments in order to convert your .asc file into a .sav file. Then, 
export as .csv, and place the file in the same directory as this script.

Output: NY_SASD_2007_CORE_CLEANED.csv
"""

# -----------------------------------------------------
# Columns available (and unavailable) for 2007 dataset.
# -----------------------------------------------------

existing_cols = ["AGE", "AGEDAY", "AGEMONTH", "AHOUR", "AYEAR", "DHOUR", "DSHOSPID", "FEMALE",
                 "HCUP_OS", "DX_Visit", "KEY", "HISPANIC",
                 "LOS", "LOS_X", "PAY1", "PAY1_X", "ORTIME", "RACE", "TOTCHG", "ANESTH", "NDX", "NPR"]

pr_cols = [f"PR{i}" for i in range(1, 16)]
prccs_cols = [f"PRCCS{i}" for i in range(1, 16)]
prday_cols = [f"PRDAY{i}" for i in range(1, 16)]
i9_dx_cols = [f"DX{i}" for i in range(1, 16)]

existing_cols.extend(
    pr_cols +
    prccs_cols +
    prday_cols +
    i9_dx_cols
)

nonexisting_cols = ["DURATION",
                    "HCUP_SURGERY_BROAD_CPT", "HCUP_SURGERY_NARROW_CPT", "HISPANIC_X", "OS_TIME",
                    "PAY2", "PAY2_X", "PAY3", "PAY3_X", "NCPT", "I10_DX_VISIT_Reason"]

cpt_cols = [f"CPT{i}" for i in range(1, 51)]
cptday_cols = [f"CPTDAY{i}" for i in range(1, 51)]
cptccs_cols = [f"CPTCCS{i}" for i in range(1, 51)]
dx_cols = [f"I10_DX{i}" for i in range(1, 16)]
cptm1_cols = [f"CPTM1_{i}" for i in range(1, 41)]
cptm2_cols = [f"CPTM2_{i}" for i in range(1, 31)]

nonexisting_cols.extend(
    cpt_cols +
    cptday_cols +
    cptccs_cols +
    dx_cols +
    cptm1_cols +
    cptm2_cols
)

# A check that makes sure all the columns in the final dataset are here.
print(len(set(existing_cols + nonexisting_cols)))

# ------------------------------------------------
# Read the dataset and make necessary adjustments.
# ------------------------------------------------

# Only including the columns that exist in this year, and converting special codes into missing values.
# Replace file path with the actual path you are using.
df = pd.read_csv("2007.csv", usecols = existing_cols, na_values = [" ", "", ".", ".A", ".B", ".C", "i", "invl", "incn", "incn2"], low_memory=False)

for col in nonexisting_cols:
    df[col] = np.nan



# Reorder columns to match final merged order. This includes all the columns needed
# to successfully merge with all the other years.
df = df[COL_ORDER]

# Convert columns to appropriate data types.
cols_to_numeric = [
    "AGE", "AGEDAY", "AGEMONTH", "AHOUR", "AYEAR",
    "DHOUR", "FEMALE", "LOS", "LOS_X",
    "ORTIME", "PAY1", "PAY2", "PAY3",
    "RACE", "HISPANIC", "HISPANIC_X",
    "TOTCHG", "HCUP_OS", "DURATION",
    "HCUP_SURGERY_BROAD_CPT", "HCUP_SURGERY_NARROW_CPT",
    "OS_TIME", "NCPT", "NPR", "NDX", "ANESTH"
]


cptday_cols = [f"CPTDAY{i}" for i in range(1, 51)]
prday_cols = [f"PRDAY{i}" for i in range(1, 16)]

cols_to_numeric.extend(cptday_cols + prday_cols)

cols_to_string = [
    "DSHOSPID", "KEY",
    "PAY1_X", "PAY2_X", "PAY3_X",
    "DX_Visit", "I10_DX_VISIT_Reason"
]

prccs_cols = [f"PRCCS{i}" for i in range(1, 16)]
pr_cols = [f"PR{i}" for i in range(1, 16)]
cptccs_cols = [f"CPTCCS{i}" for i in range(1, 51)]
cptm1_cols = [f"CPTM1_{i}" for i in range(1, 41)]
cptm2_cols = [f"CPTM2_{i}" for i in range(1, 31)]
cpt_cols = [f"CPT{i}" for i in range(1, 51)]

i9_dx_cols = [f"DX{i}" for i in range(1, 16)]
i10_dx_cols = [f"I10_DX{i}" for i in range(1, 16)]

cols_to_string.extend(
    prccs_cols +
    cptccs_cols +
    cptm1_cols +
    cptm2_cols +
    cpt_cols +
    i9_dx_cols +
    i10_dx_cols
)

for col in cols_to_numeric:
   df[col] = pd.to_numeric(df[col], errors='coerce')
for col in cols_to_string:
    df[col] = df[col].astype("string").str.strip()

# Rename diagnosis columns to match naming convention in 2008+ datasets. Plus any other necessary renaming.
df = df.rename(columns={
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
    "DX_Visit": "I9_DX_VISIT_Reason",
    "HISPANIC": "HISPANIC_X",
    "HISPANIC_X": "HISPANIC"
})

df = df[df["KEY"].notna()]

# For numeric columns, convert any negative values to actual missing values. LOS_X and PRDAYs are excluded
# as they also make use of negative values.
exclude_cols = [f"PRDAY{i}" for i in range(1, 16)] + ["LOS_X"]
cols_to_fix = [col for col in cols_to_numeric if col not in exclude_cols and col in df.columns]
df[cols_to_fix] = df[cols_to_fix].mask(df[cols_to_fix] < 0)

# Make sure each observation has a year. Remove any observations that do not have a KEY.
df["AYEAR"] = 2007
df = df[df["KEY"].notna()]

# ------------------------------------------------
# Data quality checks and export cleaned dataset.
# ------------------------------------------------

print("Dtypes:")
print(df.dtypes)
print("Missing values per column:")
print(df.isna().sum())
print(df.shape)

df.to_csv("NY_SASD_2007_CORE_CLEANED.csv", index=False)
