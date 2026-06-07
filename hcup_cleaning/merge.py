"""
Script to merge the cleaned HCUP datasets into a single file.

Input:
All of the cleaned csv files that is in the cleaned/ directory. Make sure
files are in the directory (create the directory if it doesn't exist, or change the path),
and no other csv files are present.

Output: HCUP_merged.csv
"""

import pandas as pd
import glob

files = sorted(glob.glob("cleaned/*.csv"))

output_file = "HCUP_merged.csv"
first_write = True

base_cols = None

for f in files:
    cols = list(pd.read_csv(f, nrows=0).columns)

    if base_cols is None:
        base_cols = cols
    elif cols != base_cols:
        print("Column mismatch:", f)

for f in files:
    print(f"Processing {f}...")

    for chunk in pd.read_csv(f, chunksize=100_000, low_memory=False):
        chunk.to_csv(output_file, index=False, mode="w" if first_write else "a", header=first_write)

        first_write = False

print("Done.")
