# UCI STATS 170B ENT Capstone
This is the repository for the otolaryngology data science capstone.

## DATA CLEANING
We recommend having a data folder.

### NSQIP
For NSQIP, we recommend having this folder layout:
```bash
├── ent-capstone/
    └── data/
        ├── nsqip_sav
        ├── nsqip_sav_filtered
        ├── nsqip_new
        └── nsqip
```
Keep original SAV files in **nsqip_sav**, create filtered sav files for each of those files using SPSS syntax files, and store those filtered sav files for each year to **nsqip_sav_filtered**. Use ***data_handling_nsqip.py*** to create a csv for each year in **nsqip_new**, and the combined and final filtered csv in **nsqip**. This repo includes a csv of the relevant cpt codes that we are looking at in our analysis, ***ent_cpt_codes.csv***.
