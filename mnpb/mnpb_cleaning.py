import pandas as pd
import glob
import os

# This is kinda funky to describe but within this folder you should have:
#
#   1. All the CSVs for 2009 onwards for MNPB Surgery (name as MNPB_YEAR)
#
#   2. folders for 2005-2008, since those CSVs need to be merged together first
#       named folder: YEAR_files (ex. 2008_files)
#
#ent-capstone/
#└── mnpb/
#    ├── 2005_files/
#    │   ├── Y0510040.csv
#    │   └── ...
#    ├── 2006_files/
#    │   ├── Y0620000.csv
#    │   └── ...
#    ├── 2007_files/
#    │   ├── Y07Card.csv
#    │   └── ...
#    ├── 2008_files/
#    │   ├── 20000.csv
#    │   └── ...
#    ├── MNPB_2009.csv
#    ├── MNPB_2010.csv
#    └── ... MNPB_2024.csv
#    └── MNPB_MASTER_SIMPLE.csv (output)

def find_header_row(file_path):
    """Find which row contains the actual column headers"""
    # there's usually some copyright blurb at the top of each file even after being converted to csv
    # which is necessary but also not great when trying to create a csv of the actual data
    # esp because the size and format isn't consistent over the years
    # and if you try to ingore it it'll create a bunch of nothing-columns

    df_sample = pd.read_csv(file_path, header=None, nrows=20)
    
    for idx, row in df_sample.iterrows():
        row_str = ' '.join(row.astype(str)).upper()
        # these columns are consistent and should exist throughout the whole dataset without being changed
        if ('HCPCS' in row_str and 
            ('ALLOWED SERVICES' in row_str or 'SERVICES' in row_str) and 
            ('PAYMENT' in row_str)):
            return idx
    
    # but just in case, setting row 0 as fallback
    print(f"  Warning: Could not find header row in {file_path}, assuming row 0")
    return 0

def read_mnpb_simple(file_path, year):
    """read MNPB file and extract the columns we need"""
    try:
        # Find header row
        header_row = find_header_row(file_path)
        
        df = pd.read_csv(file_path, header=header_row)
        
        #CAN COMMENT OUT just for debugging
        #print(f"  Original columns: {list(df.columns)}")
        
        # columns we want (case insensitive)
        desired_cols = ['HCPCS', 'MODIFIER', 'DESCRIPTION', 
                       'ALLOWED SERVICES', 'ALLOWED CHARGES', 'PAYMENT']
        
        # matching columns (case insensitive)
        selected_cols = {}
        for desired in desired_cols:
            for col in df.columns:
                if col.upper().strip() == desired.upper():
                    selected_cols[desired] = col
                    break
        
        # If ALLOWED SERVICES not found, try variations
        if 'ALLOWED SERVICES' not in selected_cols:
            for col in df.columns:
                if 'SERVICE' in col.upper():
                    selected_cols['ALLOWED SERVICES'] = col
                    break
        
        # If ALLOWED CHARGES not found, try variations
        if 'ALLOWED CHARGES' not in selected_cols:
            for col in df.columns:
                if 'CHARGE' in col.upper():
                    selected_cols['ALLOWED CHARGES'] = col
                    break
        
        # Create new dataframe with selected columns
        # Start with the year column
        data_dict = {'YEAR': [year] * len(df)}
        
        for desired, original in selected_cols.items():
            data_dict[desired] = df[original]
        
        new_df = pd.DataFrame(data_dict)
        
        # Remove rows where HCPCS is missing or contains copyright
        new_df = new_df[new_df['HCPCS'].notna()]
        new_df = new_df[~new_df['HCPCS'].astype(str).str.contains('copyright|Copyright', na=False)]
        
        # Remove copyright rows from DESCRIPTION if it exists
        if 'DESCRIPTION' in new_df.columns:
            new_df = new_df[~new_df['DESCRIPTION'].astype(str).str.contains('copyright|Copyright', na=False)]
        
        print(f"  Success: {len(new_df)} rows extracted")
        print(f"  Columns kept: {list(new_df.columns)}")
        print(f"  Sample YEAR values: {new_df['YEAR'].iloc[:3].tolist()}")
        
        return new_df
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

def process_all_years():
    """Process all years from 2005-2024"""
    all_data = []
    
    # Process 2005-2008 from folders
    for year in [2005, 2006, 2007, 2008]:
        folder = f'mnpb/{year}_files'
        if os.path.exists(folder):
            print(f"\nProcessing {year} from folder {folder}")
            all_files = glob.glob(os.path.join(folder, "*.csv"))
            
            for file in all_files:
                print(f"  Reading {os.path.basename(file)}")
                df = read_mnpb_simple(file, year)
                if df is not None and len(df) > 0:
                    all_data.append(df)
        else:
            print(f"\nFolder {folder} not found")
    
    # Process 2009-2024 from mnpb directory
    for year in range(2009, 2025):
        file_path = f'mnpb/MNPB_{year}.csv'
        if os.path.exists(file_path):
            print(f"\nProcessing {year} from {file_path}")
            df = read_mnpb_simple(file_path, year)
            if df is not None and len(df) > 0:
                all_data.append(df)
        else:
            print(f"\n{file_path} not found")
    
    # Combine all data
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)        
        output_file = 'mnpb/MNPB_ALL_YEARS.csv'
        master_df.to_csv(output_file, index=False)
        
        # can comment the rest of this out just for double checking
        print(f"Total rows: {len(master_df):,}")
        print(f"Columns: {list(master_df.columns)}")
        print(f"Years included: {sorted(master_df['YEAR'].unique())}")
        
        return master_df
    else:
        print("No data found")
        return None

if __name__ == "__main__":
    master_data = process_all_years()