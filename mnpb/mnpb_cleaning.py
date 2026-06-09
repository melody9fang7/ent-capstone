import pandas as pd
import glob
import os

def process_early_year_file(file_path, year):
    """Process files from 2005-2008"""
    try:
        # Find header row
        df_raw = pd.read_csv(file_path, header=None)
        header_idx = None
        
        for idx in range(len(df_raw)):
            row_str = ' '.join(str(x).upper() for x in df_raw.iloc[idx] if pd.notna(x))
            if 'HCPCS' in row_str and 'MODIFIER' in row_str and 'DESCRIPTION' in row_str:
                header_idx = idx
                break
        
        if header_idx is None:
            return None
        
        # Read with header
        df = pd.read_csv(file_path, header=header_idx)
        
        # Take first 6 columns in expected order
        df_clean = pd.DataFrame()
        df_clean['HCPCS'] = df.iloc[:, 0]
        df_clean['MODIFIER'] = df.iloc[:, 1]
        df_clean['DESCRIPTION'] = df.iloc[:, 2]
        df_clean['ALLOWED SERVICES'] = df.iloc[:, 3]
        df_clean['ALLOWED CHARGES'] = df.iloc[:, 4]
        df_clean['PAYMENT'] = df.iloc[:, 5]
        
        # Find first real data row
        start_idx = 0
        for idx in range(len(df_clean)):
            hcpcs_val = str(df_clean.iloc[idx]['HCPCS']) if pd.notna(df_clean.iloc[idx]['HCPCS']) else ''
            if hcpcs_val and hcpcs_val.strip() and '_____' not in hcpcs_val:
                start_idx = idx
                break
        
        if start_idx > 0:
            df_clean = df_clean.iloc[start_idx:].reset_index(drop=True)
        
        # Forward fill HCPCS
        last_valid = None
        for idx in range(len(df_clean)):
            current = df_clean.iloc[idx]['HCPCS']
            if pd.notna(current) and str(current).strip() and '_____' not in str(current):
                last_valid = current
            elif last_valid is not None:
                df_clean.at[idx, 'HCPCS'] = last_valid
        
        # Remove rows without HCPCS
        df_clean = df_clean[df_clean['HCPCS'].notna()].reset_index(drop=True)
        
        # Remove copyright rows
        copyright_mask = df_clean['HCPCS'].astype(str).str.contains('copyright|Copyright', na=False)
        df_clean = df_clean[~copyright_mask].reset_index(drop=True)
        
        # Extend DESCRIPTION to all rows (one description per file)
        if len(df_clean) > 0:
            first_desc = df_clean.iloc[0]['DESCRIPTION']
            if pd.notna(first_desc) and '_____' not in str(first_desc):
                df_clean['DESCRIPTION'] = first_desc
        
        # Add year
        df_clean.insert(0, 'YEAR', year)
        
        return df_clean
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

def process_recent_year_file(file_path, year):
    """Process files from 2009 onwards - HCPCS and DESCRIPTION need forward filling"""
    try:
        df = pd.read_csv(file_path)
        
        # Columns in order: DESCRIPTION, HCPCS, MODIFIER, ALLOWED SERVICES, ALLOWED CHARGES, PAYMENT
        df_clean = pd.DataFrame()
        df_clean['DESCRIPTION'] = df.iloc[:, 0]
        df_clean['HCPCS'] = df.iloc[:, 1]
        df_clean['MODIFIER'] = df.iloc[:, 2]
        df_clean['ALLOWED SERVICES'] = df.iloc[:, 3]
        df_clean['ALLOWED CHARGES'] = df.iloc[:, 4]
        df_clean['PAYMENT'] = df.iloc[:, 5]
        
        # Remove copyright rows
        copyright_mask = df_clean['HCPCS'].astype(str).str.contains('copyright|Copyright', na=False)
        df_clean = df_clean[~copyright_mask].reset_index(drop=True)
        
        # Forward fill DESCRIPTION (extend each description down until next non-null)
        last_valid_desc = None
        for idx in range(len(df_clean)):
            current_desc = df_clean.iloc[idx]['DESCRIPTION']
            if pd.notna(current_desc) and str(current_desc).strip():
                last_valid_desc = current_desc
            elif last_valid_desc is not None:
                df_clean.at[idx, 'DESCRIPTION'] = last_valid_desc
        
        # Forward fill HCPCS (extend each HCPCS down until next non-null)
        # This handles the case where HCPCS only appears on TOTAL row
        last_valid_hcpcs = None
        for idx in range(len(df_clean)):
            current_hcpcs = df_clean.iloc[idx]['HCPCS']
            if pd.notna(current_hcpcs) and str(current_hcpcs).strip():
                last_valid_hcpcs = current_hcpcs
            elif last_valid_hcpcs is not None:
                df_clean.at[idx, 'HCPCS'] = last_valid_hcpcs
        
        # Remove rows that still don't have HCPCS
        df_clean = df_clean[df_clean['HCPCS'].notna()].reset_index(drop=True)
        
        # Convert ALLOWED SERVICES to numeric (remove $ and commas if present)
        if 'ALLOWED SERVICES' in df_clean.columns:
            df_clean['ALLOWED SERVICES'] = df_clean['ALLOWED SERVICES'].astype(str)
            df_clean['ALLOWED SERVICES'] = df_clean['ALLOWED SERVICES'].str.replace('$', '', regex=False)
            df_clean['ALLOWED SERVICES'] = df_clean['ALLOWED SERVICES'].str.replace(',', '', regex=False)
            df_clean['ALLOWED SERVICES'] = pd.to_numeric(df_clean['ALLOWED SERVICES'], errors='coerce')
        
        # Add year
        df_clean.insert(0, 'YEAR', year)
        
        # Reorder to match early years
        df_clean = df_clean[['YEAR', 'HCPCS', 'MODIFIER', 'DESCRIPTION', 'ALLOWED SERVICES', 'ALLOWED CHARGES', 'PAYMENT']]
        
        print(f"  Processed {len(df_clean)} rows for {year}")
        return df_clean
        
    except Exception as e:
        print(f"  Error for {year}: {e}")
        return None

def process_all_years():
    """Process all years and create master CSV"""
    all_data = []
    
    # Process 2005-2008
    for year in [2005, 2006, 2007, 2008]:
        folder = f'mnpb/{year}_files'
        if os.path.exists(folder):
            print(f"\nProcessing {year}...")
            files = glob.glob(os.path.join(folder, "*.csv"))
            for f in files:
                print(f"  {os.path.basename(f)}")
                df = process_early_year_file(f, year)
                if df is not None and len(df) > 0:
                    all_data.append(df)
    
    # Process 2009-2024
    print(f"\nProcessing 2009-2024...")
    for year in range(2009, 2025):
        file_path = f'mnpb/2009_onwards/MNPB_{year}.csv'
        if os.path.exists(file_path):
            df = process_recent_year_file(file_path, year)
            if df is not None and len(df) > 0:
                all_data.append(df)
        else:
            print(f"  {year} not found")
    
    # Combine and save
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        master_df.to_csv('mnpb/MNPB_MASTER_FINAL.csv', index=False)
        
        print(f"Saved to mnpb/MNPB_MASTER_FINAL.csv")
        print(f"Total rows: {len(master_df):,}")
        print(f"Years: {sorted(master_df['YEAR'].unique())}")
        
        # Verify 2009
        if 2009 in master_df['YEAR'].unique():
            year_data = master_df[master_df['YEAR'] == 2009]
            print(f"\n2009 Verification:")
            print(f"  Rows: {len(year_data):,}")
            print(f"  Unique HCPCS: {year_data['HCPCS'].nunique()}")
            print(f"  Unique DESCRIPTION: {year_data['DESCRIPTION'].nunique()}")
            print(f"  Unique MODIFIER: {year_data['MODIFIER'].unique().tolist()}")
            print(f"  Sample HCPCS: {year_data['HCPCS'].head(5).tolist()}")
            print(f"  Sample MODIFIER: {year_data['MODIFIER'].head(5).tolist()}")
        
        print(f"{'='*50}")
        return master_df
    else:
        print("No data found!")
        return None

if __name__ == "__main__":
    master_data = process_all_years()