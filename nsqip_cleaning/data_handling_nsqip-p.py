import pandas as pd
import glob
import os

# need to add PUFYEAR to pre-2015 files
def fix_pufyear_ped(): # if youre running this -- make sure 2012-2014 are named as such (or swap out for what theyre named on your end)
    puf_year_csvs = {'acs_nsqipp_puf12.csv': 2012, 'acs_nsqipp_puf13.csv': 2013, 'acs_nsqipp_puf14.csv': 2014}

    for key, val in puf_year_csvs.items():
        df = pd.read_csv(key)
        df.insert(0, 'PUFYEAR', val)
        df.to_csv(key, index=False)

# ANESTHES is called ANESTECH in some nsqipp files
def fix_anesthes(PATH):
    for file in glob.glob(PATH):
        df = pd.read_csv(file)
        df = df.rename(columns={'ANESTECH': 'ANESTHES'})    
        df.to_csv(file, index=False)

_RACE_KEYWORDS = {
    'White':                              'White',
    'Black or African American':          'Black',
    'Black':                              'Black',
    'Asian':                              'Asian',
    'American Indian or Alaska Native':   'AIAN',
    'Native Hawaiian':                    'NHPI',
    'Some Other Race':                    'Other',
    'Race Combinations with Low Frequency':             'Other',
    'Native Hawaiian or Other Pacific Islander':        'NHPI',
    'Unknown/Not Reported:':                            'Unknown/Not Reported',
    'White,Some Other Race':                            'Other',
    'White,Asian':                         'Other',
    'White,Black or African American':      'Other',
    'Race combinations with low frequency':  'Other'
}

def fix_races(PATH):
    for file in glob.glob(PATH):
        df = pd.read_csv(file)
        
        # renaming column if 'RACE_NEW' exists, otherwise use 'RACE'
        if 'RACE_NEW' in df.columns:
            df = df.rename(columns={'RACE_NEW': 'RACE'})
        
        if 'RACE' in df.columns:
            df['RACE'] = df['RACE'].map(_RACE_KEYWORDS).fillna(df['RACE'])
            
            unmapped = df[~df['RACE'].isin(set(_RACE_KEYWORDS.values()))]['RACE'].unique()
            if len(unmapped) > 0:
                print(f"\nNote in {file}: Un-mapped Values:")
                for val in unmapped:
                    count = df[df['RACE'] == val].shape[0]
                    print(f"  '{val}': {count} occurrences")
            
            df.to_csv(file, index=False)
            print(f"\nProcessed: {file}")
        else:
            print(f"\nWarning: No 'RACE' or 'RACE_NEW' column found in {file}")

def standardize_all(PATH):
    """Standardize CPT, CASEID, and HISPANIC"""
    for file in glob.glob(PATH):
        df = pd.read_csv(file)
        
        if 'CPT' in df.columns:
            df['CPT'] = df['CPT'].astype('Int64')
        
        if 'CASEID' in df.columns:
            df['CASEID'] = df['CASEID'].astype('Int64')
        
        if 'ETHNICITY_HISPANIC' in df.columns:
            df['ETHNICITY_HISPANIC'] = df['ETHNICITY_HISPANIC'].replace({
                'YES': 'Y', 'Yes': 'Y', 'yes': 'Y', 'Y': 'Y', '1': 'Y', 1: 'Y',
                'NO': 'N', 'No': 'N', 'no': 'N', 'N': 'N', '0': 'N', 0: 'N',
                'NULL': None, 'null': None, 'None': None, '': None
                })
        
        df.to_csv(file, index=False)
        print(f"Standardized: {file}")

def merge_csvs(PATH, output_file="ALL_NSQIP-P.csv"):
    all_dfs = []
    
    for file in glob.glob(PATH):
        df = pd.read_csv(file)
        all_dfs.append(df)
        print(f"Loaded {file}: {len(df)} rows")
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    
    return merged_df


if __name__ == "__main__":    
    PATH = '/Users/user/Desktop/CAPSTONE/NSQIP-P/nsqipp csv/*.csv' # insert your own path here

    #fix_pufyear_ped() # CHECK WHAT YOUR 2012-2014 FILES ARE NAMED IN FUNCTION 
    #fix_anesthes(PATH)
    #fix_races(PATH)
    #standardize_all(PATH)
    #merge_csvs(PATH)