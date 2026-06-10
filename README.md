# UCI STATS 170B ENT Capstone

## Project Overview

This project analyzes otolaryngology procedure outcomes using NSQIP and HCUP datasets. The workflow consists of:

1. Cleaning and standardizing raw NSQIP and HCUP data.
2. Filtering records to ENT-related CPT codes.
3. Creating merged analysis datasets.
4. Performing statistical analyses, including segmented regression, mixed-effects models, volume analyses, and causal inference methods.
5. Generating visualizations and summary statistics.

## Repository Structure

### Root Directory

| File                        | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| `README.md`                 | Project documentation and usage instructions.        |
| `ipynb_data_generator.py`   | Generate sample data for jupyter notebook,           |
|                             |    (ONLY FOR ILLUSTRATIVE PURPOSES)                  |
| `notebook.ipynb`            | Illustrates modeling.                                |
| `notebook.html`             | Html output of jupyter notebook                      |

### NSQIP Cleaning (`nsqip_cleaning/`)

| File                        | Description                                     |
| --------------------------- | ----------------------------------------------- |
| `data_handling_nsqip.py`    | Cleans and standardizes NSQIP datasets.         |
| `data_handling_nsqip-p.py`  | Cleans and standardizes NSQIP-P datasets.       |
| `cpt_codes.py`              | Original code to find and analyze available CPTs.|
| `stats.py`                  | NSQIP statistical analysis and linear regression.|
| `mixed_effects.py`          | Mixed-effects regression analysis.              |
| `nsqip_segmented_lr.py`     | Segmented regression analysis.                  |
| `nsqip_segmented_volume.py` | Volume-based segmented analysis.                |
| `causalforest.py`           | Causal forest treatment-effect analysis.        |

### HCUP Cleaning (`hcup_cleaning/`)

| File                            | Description                                |
| ------------------------------- | ------------------------------------------ |
| `clean2007.py` – `clean2017.py` | Year-specific HCUP cleaning scripts.       |
| `column_order.py`               | Standardizes column ordering across years. |
| `filtering.py`                  | Filters records to the study population.   |
| `merge.py`                      | Merges cleaned HCUP datasets.              |
| `hcup_stats.py`                 | HCUP statistical analysis and linear regression.    |
| `mixed_effects_hcup.py`         | Mixed-effects modeling on HCUP data.       |
| `segmented_hcup.py`             | Segmented regression analysis.             |
| `segmented_hcup_volume.py`      | Volume-based segmented analysis.           |
| `volume_analysis.py`            | Procedure volume analysis.                 |

### MNPB Cleaning (`mnpb/`)
| File                            | Description                                |
| ------------------------------- | ------------------------------------------ |
| `mnpb_viz.py`                   | Visualize volumes w/o segmented reg.       |
| `mnpb_volume_segreg.py`         | Volume Segmented regression MNPB data.     |
| `mnpb_cleaning.py`              | MNPB Cleaning.                             |

## Data Requirements

This project requires access to:

* NSQIP datasets
* HCUP datasets
* ENT CPT code reference files

These datasets are not included in the repository due to licensing restrictions. We have included mockups of what the cleaned data would like inside the sample_data folder. 

## Recommended Directory Structure

```text
ent-capstone/
├── sample_data/
├── data/
│   ├── nsqip_sav/
│   ├── nsqip_sav_filtered/
│   ├── nsqip_new/
│   ├── nsqip/
│   ├── hcup_raw/
│   ├── hcup_cleaned/
│   └── hcup_merged/
├── nsqip_cleaning/
├── hcup_cleaning/
├── mnpb/
├── ipynb_data_generator.py
├── notebook.ipynb
└── sample_data/
```

## Running the Pipeline

### Step 1: Clean NSQIP Data

```bash
python nsqip_cleaning/data_handling_nsqip.py
```

This creates cleaned yearly CSV files and a combined NSQIP dataset.

### Step 2: Clean HCUP Data

Run each yearly cleaning script:

```bash
python hcup_cleaning/clean2007.py
python hcup_cleaning/clean2008.py
...
python hcup_cleaning/clean2017.py
```

### Step 3: Merge HCUP Data

```bash
python hcup_cleaning/merge.py
```

### Step 4: Generate Analysis Results

Examples:

```bash
python nsqip_cleaning/stats.py
python nsqip_cleaning/mixed_effects.py
python nsqip_cleaning/nsqip_segmented_lr.py
python hcup_cleaning/mixed_effects_hcup.py
python hcup_cleaning/segmented_hcup.py
```
## Manual Steps

1. Obtain licensed NSQIP and HCUP datasets.
2. Place raw files in the appropriate data directories.
3. Apply any required SPSS syntax filtering to NSQIP SAV files.
4. Run the cleaning scripts in chronological order.
5. Merge cleaned datasets.
6. Run statistical analyses.
7. Generate final tables and figures.

## Final Outputs

The pipeline produces:

* Cleaned NSQIP datasets
* Cleaned HCUP datasets
* Statistical model outputs
* Segmented regression results
* Volume analyses
* Figures and visualizations used in the final report
