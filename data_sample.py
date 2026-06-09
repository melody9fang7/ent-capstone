import pandas as pd
import numpy as np
from faker import Faker
import random

VALID_CPTS = [
    '60240', '60220', '38724', '42415', '60252',
    '42145', '60260', '42440', '42420', '41120',
    '60271', '60254', '31360', '41135', '38720'
]

CPT_GROUPS = {
    '60240': 'Thyroid Surgeries', '60220': 'Thyroid Surgeries',
    '60252': 'Thyroid Surgeries', '60254': 'Thyroid Surgeries',
    '60260': 'Thyroid Surgeries', '60271': 'Thyroid Surgeries',
    '38724': 'Neck Dissections', '38720': 'Neck Dissections',
    '42415': 'Salivary Gland Surgeries', '42440': 'Salivary Gland Surgeries',
    '42420': 'Salivary Gland Surgeries', '42145': 'Miscellaneous Codes',
    '41120': 'Glossectomies and Laryngectomies',
    '41135': 'Glossectomies and Laryngectomies',
    '31360': 'Glossectomies and Laryngectomies',
}

WORK_RVUS = {
    '60240': 15.04, '60220': 11.19, '38724': 14.52, '42415': 16.86,
    '60252': 22.01, '42145': 9.78, '60260': 18.26, '42440': 6.14,
    '42420': 19.53, '41120': 11.14, '60271': 17.62, '60254': 28.42,
    '31360': 29.91, '41135': 30.14, '38720': 21.95
}

OPTIME_PARAMS = {
    '60240': (150, 40), '60220': (90, 25), '38724': (180, 50),
    '42415': (150, 45), '60252': (180, 55), '42145': (60, 20),
    '60260': (145, 40), '42440': (60, 20), '42420': (180, 50),
    '41120': (60, 20), '60271': (150, 40), '60254': (210, 60),
    '31360': (200, 60), '41135': (210, 65), '38720': (150, 45)
}

SURGSPEC_OPTIONS = ['Otolaryngology (ENT)']
ANESTHES_OPTIONS = ['General', 'Spinal', 'Epidural', 'MAC/IV Sedation', 'Local']
ASACLAS_OPTIONS = [1, 2, 3, 4]
RACE_OPTIONS = ['White', 'Black or African American', 'Asian', 'Unknown/Not Reported']
CASETYPE_OPTIONS = ['Elective', 'Urgent', 'Emergent']


def generate_fake_nsqip(source_file_path: str, target_file_path: str, num_rows: int = 100):
    fake = Faker()
    fake_data = []

    print("generating rows...")
    for i in range(num_rows):
        cpt = random.choice(VALID_CPTS)
        year = random.randint(2005, 2022)
        age_raw = random.randint(18, 95)
        age = '90+' if age_raw > 90 else str(age_raw)

        # realistic operative time from CPT-specific distribution
        optime_mean, optime_std = OPTIME_PARAMS[cpt]
        optime = max(10, int(np.random.normal(optime_mean, optime_std)))

        # most cases are solo — ~75% chance no co-occurring procedures
        has_other = random.random() > 0.75
        other_cpt = random.choice(VALID_CPTS) if has_other else None

        row = {
            'PUFYEAR':              year,
            'CASEID':               fake.unique.random_int(min=100000, max=999999),
            'SEX':                  random.choice(['female', 'male']),
            'PRNCPTX':              f'PROCEDURE {cpt}',
            'CPT':                  cpt,
            'WORKRVU':              WORK_RVUS[cpt],
            'AGE':                  age,
            'ANESTHES':             random.choice(ANESTHES_OPTIONS),
            'SURGSPEC':             'Otolaryngology (ENT)',
            'OTHERPROC1':           fake.word() if has_other else None,
            'OTHERPROC2':           None,
            'OTHERPROC3':           None,
            'OTHERPROC4':           None,
            'OTHERPROC5':           None,
            'OTHERPROC6':           None,
            'OTHERPROC7':           None,
            'OTHERPROC8':           None,
            'OTHERPROC9':           None,
            'OTHERPROC10':          None,
            'OTHERCPT1':            other_cpt,
            'OTHERCPT2':            None,
            'OTHERCPT3':            None,
            'OTHERCPT4':            None,
            'OTHERCPT5':            None,
            'OTHERCPT6':            None,
            'OTHERCPT7':            None,
            'OTHERCPT8':            None,
            'OTHERCPT9':            None,
            'OTHERCPT10':           None,
            'OTHERWRVU1':           WORK_RVUS[other_cpt] if other_cpt else None,
            'OTHERWRVU2':           None,
            'OTHERWRVU3':           None,
            'OTHERWRVU4':           None,
            'OTHERWRVU5':           None,
            'OTHERWRVU6':           None,
            'OTHERWRVU7':           None,
            'OTHERWRVU8':           None,
            'OTHERWRVU9':           None,
            'OTHERWRVU10':          None,
            'CONCURR1':             None,
            'CONCURR2':             None,
            'CONCURR3':             None,
            'CONCURR4':             None,
            'CONCURR5':             None,
            'CONCURR6':             None,
            'CONCURR7':             None,
            'CONCURR8':             None,
            'CONCURR9':             None,
            'CONCURR10':            None,
            'CONCPT1':              None,
            'CONCPT2':              None,
            'CONCPT3':              None,
            'CONCPT4':              None,
            'CONCPT5':              None,
            'CONCPT6':              None,
            'CONCPT7':              None,
            'CONCPT8':              None,
            'CONCPT9':              None,
            'CONCPT10':             None,
            'CONWRVU1':             None,
            'CONWRVU2':             None,
            'CONWRVU3':             None,
            'CONWRVU4':             None,
            'CONWRVU5':             None,
            'CONWRVU6':             None,
            'CONWRVU7':             None,
            'CONWRVU8':             None,
            'CONWRVU9':             None,
            'CONWRVU10':            None,
            'ASACLAS':              random.choice(ASACLAS_OPTIONS),
            'OPTIME':               optime,
            'TOTHLOS':              random.randint(0, 7),
            'DOPTODIS':             random.randint(0, 7),
            'CPT GROUP':            CPT_GROUPS[cpt],
            'RACE_NEW':             random.choice(RACE_OPTIONS),
            'ETHNICITY_HISPANIC':   random.choice(['Yes', 'No', 'Unknown']),
            'ELECTSURG':            random.choice(['Y', 'N']),
            'CASETYPE':             random.choice(CASETYPE_OPTIONS),
        }

        fake_data.append(row)
        if i % 10 == 0:
            print(f"  {i}/{num_rows}")

    df_fake = pd.DataFrame(fake_data)
    df_fake.to_csv(target_file_path, index=False)
    print(f"saved {num_rows} rows to {target_file_path}")


if __name__ == "__main__":
    generate_fake_nsqip(
        source_file_path="data/nsqip/combined_filtered.csv",
        target_file_path="sample_data/nsqip_sample_data.csv",
        num_rows=100
    )