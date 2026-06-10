import pandas as pd
import numpy as np
from faker import Faker
import random

ICD10_OPTIONS = ['J342', 'J343','J3489', 'G4733', 'M950',]

ICD9_OPTIONS = ['470', '4780', '47819', '32723', '7380',]

# one for now
CPT_GROUPS = {'30520': 'Septoplasty/Turbinectomy/Closed Nasal Bone Reduction'}

OTHER_CPTS = [
    '30140', '30801', '30802',
    '30930', '21320', '31237',
    '31256'
]

RACE_OPTIONS = [1, 2, 3, 4, 5, 6]

def generate_fake_hcup(target_file_path, num_rows=1000):
    fake = Faker()
    fake_data = []

    print("generating rows...")

    for i in range(num_rows):
        dx9_idx = random.randint(0, len(ICD9_OPTIONS) - 1)
        dx10_idx = random.randint(0, len(ICD10_OPTIONS) - 1)

        los = random.choice([0, 0, 0, 0, 0, 0, 0, 1])

        # ORTIME based on CPT 30520
        ortime = max(10, int(np.random.normal(80, 53)))

        # most cases are solo — ~75% chance no co-occurring procedure
        has_other = random.random() > 0.75
        other_cpt = random.choice(OTHER_CPTS) if has_other else np.nan
        ncpt = 2 if has_other else 1

        row = {
            'AGE': random.randint(14, 103),
            'AGEDAY': np.nan,
            'AGEMONTH': np.nan,
            'AHOUR': random.choice([600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]),
            'ANESTH': random.choice([0, 10, 20, 30, 40]),
            'AYEAR': random.randint(2008, 2017),
            'CPT1': '30520',
            'CPT2': other_cpt,
            'CPT3': np.nan,
            'CPT4': np.nan,
            'CPT5': np.nan,
            'CPT6': np.nan,
            'CPT7': np.nan,
            'CPT8': np.nan,
            'CPT9': np.nan,
            'CPT10': np.nan,
            'CPTCCS1': 28, # gspecific to 30520
            'CPTCCS2': np.nan,
            'CPTCCS3': np.nan,
            'CPTCCS4': np.nan,
            'CPTCCS5': np.nan,
            'CPTCCS6': np.nan,
            'CPTCCS7': np.nan,
            'CPTCCS8': np.nan,
            'CPTCCS9': np.nan,
            'CPTCCS10': np.nan,
            'CPTDAY1': np.nan,
            'CPTDAY2': np.nan,
            'CPTM1_1': random.choice([np.nan, np.nan, np.nan, np.nan, 'LT', 'RT', '50', '51']),
            'CPTM2_1': np.nan,
            'DHOUR': random.choice([1000, 1100, 1200, 1300, 1400, 1500]),
            'DSHOSPID': random.choice([1, 2, random.randint(100, 299)]),
            'DURATION': np.nan,
            'FEMALE': random.choice([0, 1]),
            'HCUP_OS': 0,
            'HCUP_SURGERY_BROAD_CPT': np.nan,
            'HCUP_SURGERY_NARROW_CPT': np.nan,
            'HISPANIC': random.choice([0, 1, 2, 3, 4, np.nan]),
            'HISPANIC_X': np.nan,
            'I10_DX1': ICD10_OPTIONS[dx10_idx],
            'I10_DX2': np.nan,
            'I10_DX3': np.nan,
            'I10_DX4': np.nan,
            'I10_DX5': np.nan,
            'I10_DX_VISIT_Reason': ICD10_OPTIONS[dx10_idx],
            'I9_DX1': ICD9_OPTIONS[dx9_idx],
            'I9_DX2': np.nan,
            'I9_DX3': np.nan,
            'I9_DX4': np.nan,
            'I9_DX5': np.nan,
            'I9_DX_VISIT_Reason': ICD9_OPTIONS[dx9_idx],
            'KEY': fake.unique.random_int(min=300000000, max=999999999),
            'LOS': los,
            'LOS_X': los,
            'NCPT': ncpt,
            'NDX': 1,
            'ORTIME': ortime,
            'OS_TIME': np.nan,
            'PAY1': random.choice([np.nan, np.nan, np.nan, np.nan, 1, 2, 3, 4, 6]),
            'PAY1_X': np.nan,
            'PAY2': random.choice([np.nan, np.nan, np.nan, np.nan, 1, 2, 3, 4, 6]),
            'PAY2_X': np.nan,
            'PAY3': random.choice([np.nan, np.nan, np.nan, np.nan, 1, 2, 3, 4, 6]),
            'PAY3_X': np.nan,
            'RACE': random.choice(RACE_OPTIONS),
            'TOTCHG': random.randint(300,15000)
        }

        fake_data.append(row)

        if i % 10 == 0:
            print(f"{i}/{num_rows}")

    df_fake = pd.DataFrame(fake_data)
    df_fake.to_csv(target_file_path, index=False )
    print(f"saved {num_rows} rows to {target_file_path}")

def generate_fake_nsqip(target_file_path: str = 'nsqip_sample_data.csv', num_rows: int = 500000):
    """
    Generate fake NSQIP data with multiple CPTs.
    CPT 12345 has wRVU changes AND volume decreases after the 2012 devaluation.
    """
    np.random.seed(42)
    
    TARGET_CPT = '12345'
    OTHER_CPTS = ['50001', '50002', '50003', '50004', '50005']
    
    rows = []
    for i in range(num_rows):
        year = np.random.randint(2005, 2023)
        
        # Volume share for target CPT: stable ~10% before 2012, drops to ~7% after
        if year < 2007:
            target_share = 0.10
            wrvu = 10.0
        elif year < 2010:
            target_share = 0.10
            wrvu = 11.0   # +10% in 2007
        elif year < 2012:
            target_share = 0.10
            wrvu = 11.55  # +5% in 2010
        else:
            target_share = 0.07  # Volume drops after 2012 devaluation
            wrvu = 9.82   # -15% in 2012
        
        if np.random.random() < target_share:
            cpt = TARGET_CPT
            optime_mean = 120
        else:
            cpt = np.random.choice(OTHER_CPTS)
            wrvu = round(np.random.uniform(5, 20), 2)
            optime_mean = np.random.uniform(60, 200)
        
        optime = max(10, int(np.random.normal(optime_mean, 30)))
        
        rows.append({
            'PUFYEAR': year,
            'CASEID': 100000 + i,
            'CPT': cpt,
            'WORKRVU': wrvu,
            'OPTIME': optime,
            'SEX': np.random.choice(['female', 'male']),
            'AGE': np.random.randint(18, 91),
            'ASACLAS': np.random.choice([1, 2, 3, 4]),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(target_file_path, index=False)

def generate_fake_mixed_data(target_file='sample_data/nsqip_mixed_sample.csv', num_rows=2000):
    """Generate balanced fake panel data for mixed-effects demo."""
    np.random.seed(42)
    
    # 20 increases, 10 decreases — more balanced
    inc_cpts = [f'50{i:03d}' for i in range(20)]
    dec_cpts = [f'50{i:03d}' for i in range(20, 30)]
    all_cpts = inc_cpts + dec_cpts
    reval_year = 2012
    
    rows = []
    for i in range(num_rows):
        cpt = np.random.choice(all_cpts)
        is_dec = 1 if cpt in dec_cpts else 0
        year = np.random.randint(2006, 2023)
        
        # All CPTs have similar baseline volume
        base_vol = np.random.uniform(0.04, 0.10)
        
        # Decreases: modest 15% drop after 2012
        if is_dec and year >= reval_year:
            base_vol *= 0.85
        
        # Same noise for all
        volume = base_vol + np.random.normal(0, 0.008)
        
        rows.append({
            'cpt': cpt,
            'year': year,
            'year_c': year - reval_year,
            'post': 1 if year >= reval_year else 0,
            'volume_rate': max(0.001, volume),
            'is_decrease': is_dec,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(target_file, index=False)

if __name__ == "__main__":
    generate_fake_hcup("sample_data/hcup_sample_data.csv", 1000)
    generate_fake_nsqip('nsqip_sample_data.csv', num_rows=500_000) # maybe change lol
    generate_fake_mixed_data()
