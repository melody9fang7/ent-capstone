import pandas as pd
import numpy as np

input_file = "NY_SASD_2012_CORE.csv"
output_file = "NY_SASD_2012_CORE_CLEANED.csv"
chunksize = 100_000
print("Running...")

col_order = ['AGE', 'AGEDAY', 'AGEMONTH', 'AHOUR', 'ANESTH', 'AYEAR',

 'CPT1','CPT2','CPT3','CPT4','CPT5','CPT6','CPT7','CPT8','CPT9','CPT10',
 'CPT11','CPT12','CPT13','CPT14','CPT15','CPT16','CPT17','CPT18','CPT19','CPT20',
 'CPT21','CPT22','CPT23','CPT24','CPT25','CPT26','CPT27','CPT28','CPT29','CPT30',
 'CPT31','CPT32','CPT33','CPT34','CPT35','CPT36','CPT37','CPT38','CPT39','CPT40',
 'CPT41','CPT42','CPT43','CPT44','CPT45','CPT46','CPT47','CPT48','CPT49','CPT50',

 'CPTCCS1','CPTCCS2','CPTCCS3','CPTCCS4','CPTCCS5','CPTCCS6','CPTCCS7','CPTCCS8','CPTCCS9','CPTCCS10',
 'CPTCCS11','CPTCCS12','CPTCCS13','CPTCCS14','CPTCCS15','CPTCCS16','CPTCCS17','CPTCCS18','CPTCCS19','CPTCCS20',
 'CPTCCS21','CPTCCS22','CPTCCS23','CPTCCS24','CPTCCS25','CPTCCS26','CPTCCS27','CPTCCS28','CPTCCS29','CPTCCS30',
 'CPTCCS31','CPTCCS32','CPTCCS33','CPTCCS34','CPTCCS35','CPTCCS36','CPTCCS37','CPTCCS38','CPTCCS39','CPTCCS40',
 'CPTCCS41','CPTCCS42','CPTCCS43','CPTCCS44','CPTCCS45','CPTCCS46','CPTCCS47','CPTCCS48','CPTCCS49','CPTCCS50',

 'CPTDAY1','CPTDAY2','CPTDAY3','CPTDAY4','CPTDAY5','CPTDAY6','CPTDAY7','CPTDAY8','CPTDAY9','CPTDAY10',
 'CPTDAY11','CPTDAY12','CPTDAY13','CPTDAY14','CPTDAY15','CPTDAY16','CPTDAY17','CPTDAY18','CPTDAY19','CPTDAY20',
 'CPTDAY21','CPTDAY22','CPTDAY23','CPTDAY24','CPTDAY25','CPTDAY26','CPTDAY27','CPTDAY28','CPTDAY29','CPTDAY30',
 'CPTDAY31','CPTDAY32','CPTDAY33','CPTDAY34','CPTDAY35','CPTDAY36','CPTDAY37','CPTDAY38','CPTDAY39','CPTDAY40',
 'CPTDAY41','CPTDAY42','CPTDAY43','CPTDAY44','CPTDAY45','CPTDAY46','CPTDAY47','CPTDAY48','CPTDAY49','CPTDAY50',

 'CPTM1_1','CPTM1_2','CPTM1_3','CPTM1_4','CPTM1_5','CPTM1_6','CPTM1_7','CPTM1_8','CPTM1_9','CPTM1_10',
 'CPTM1_11','CPTM1_12','CPTM1_13','CPTM1_14','CPTM1_15','CPTM1_16','CPTM1_17','CPTM1_18','CPTM1_19','CPTM1_20',
 'CPTM1_21','CPTM1_22','CPTM1_23','CPTM1_24','CPTM1_25','CPTM1_26','CPTM1_27','CPTM1_28','CPTM1_29','CPTM1_30',
 'CPTM1_31','CPTM1_32','CPTM1_33','CPTM1_34','CPTM1_35','CPTM1_36','CPTM1_37','CPTM1_38','CPTM1_39','CPTM1_40',

 'CPTM2_1','CPTM2_2','CPTM2_3','CPTM2_4','CPTM2_5','CPTM2_6','CPTM2_7','CPTM2_8','CPTM2_9','CPTM2_10',
 'CPTM2_11','CPTM2_12','CPTM2_13','CPTM2_14','CPTM2_15','CPTM2_16','CPTM2_17','CPTM2_18','CPTM2_19','CPTM2_20',
 'CPTM2_21','CPTM2_22','CPTM2_23','CPTM2_24','CPTM2_25','CPTM2_26','CPTM2_27','CPTM2_28','CPTM2_29','CPTM2_30',

 'DHOUR','DSHOSPID','DURATION','FEMALE','HCUP_OS','HCUP_SURGERY_BROAD_CPT','HCUP_SURGERY_NARROW_CPT',
 'HISPANIC','HISPANIC_X',

 'I10_DX1','I10_DX2','I10_DX3','I10_DX4','I10_DX5','I10_DX6','I10_DX7','I10_DX8','I10_DX9','I10_DX10',
 'I10_DX11','I10_DX12','I10_DX13','I10_DX14','I10_DX15','I10_DX_VISIT_Reason',

 'I9_DX1','I9_DX2','I9_DX3','I9_DX4','I9_DX5','I9_DX6','I9_DX7','I9_DX8','I9_DX9','I9_DX10',
 'I9_DX11','I9_DX12','I9_DX13','I9_DX14','I9_DX15','I9_DX_VISIT_Reason',

 'KEY','LOS','LOS_X','NCPT','NDX','NPR','ORTIME','OS_TIME',
 'PAY1','PAY1_X','PAY2','PAY2_X','PAY3','PAY3_X',

 'PR1','PR2','PR3','PR4','PR5','PR6','PR7','PR8','PR9','PR10',
 'PR11','PR12','PR13','PR14','PR15',

 'PRCCS1','PRCCS2','PRCCS3','PRCCS4','PRCCS5','PRCCS6','PRCCS7','PRCCS8','PRCCS9','PRCCS10',
 'PRCCS11','PRCCS12','PRCCS13','PRCCS14','PRCCS15',

 'PRDAY1','PRDAY2','PRDAY3','PRDAY4','PRDAY5','PRDAY6','PRDAY7','PRDAY8','PRDAY9','PRDAY10',
 'PRDAY11','PRDAY12','PRDAY13','PRDAY14','PRDAY15',

 'RACE','TOTCHG']

existing_cols = ["AGE", "AGEDAY", "AGEMONTH", "AHOUR", "AYEAR", "DHOUR", "DSHOSPID", "DURATION", "FEMALE",
                 "HCUP_OS", "DX_Visit_Reason1", "KEY", "ANESTH", "HISPANIC_X", 
                 "LOS", "LOS_X", "NCPT", "PAY1", "PAY1_X", "ORTIME", "RACE", "TOTCHG", "NDX"]

existing_cols.extend(
    [f"CPT{i}" for i in range(1, 51)] +
    [f"CPTCCS{i}" for i in range(1, 51)] +
    [f"DX{i}" for i in range(1, 16)] +
    [f"CPTM1_{i}" for i in range(1, 41)] +
    [f"CPTM2_{i}" for i in range(1, 31)]
)

nonexisting_cols = ["HISPANIC", "OS_TIME", "HCUP_SURGERY_BROAD_CPT", "HCUP_SURGERY_NARROW_CPT",
                    "PAY2", "PAY2_X", "PAY3", "PAY3_X", "I10_DX_VISIT_Reason", "NPR"]


nonexisting_cols.extend(
    [f"CPTDAY{i}" for i in range(1, 51)] +
    [f"PR{i}" for i in range(1, 16)] +
    [f"PRCCS{i}" for i in range(1, 16)] +
    [f"PRDAY{i}" for i in range(1, 16)] +
    [f"I10_DX{i}" for i in range(1, 16)] 
)

print(len(set(existing_cols + nonexisting_cols)))


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
    "DX_Visit_Reason1", "I10_DX_VISIT_Reason"
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

    chunk["AYEAR"] = 2012

    chunk = chunk[col_order]

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
