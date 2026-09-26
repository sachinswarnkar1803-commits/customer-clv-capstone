# Raw Data Directory

This folder stores the raw data for the BDS-34 Capstone Project.

## Dataset Provenance
- **Dataset Name**: Online Retail II
- **Official Source**: UCI Machine Learning Repository
- **URL**: [https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii)
- **Direct Download URL**: `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`
- **Donor**: Dr. Daqing Chen, Director of Public Analytics group, London South Bank University
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Timeframe Covered**: 01/12/2009 to 09/12/2011 (approx. 2 years)

## Expected Files
- `online_retail_II.xlsx` (contains two sheets: `Year 2009-2010` and `Year 2010-2011`)
- Or cached raw CSV extracts: `online_retail_II_raw.csv`

## Ingestion
The ingestion script `src/data/ingest.py` automatically checks for these files and can download them directly from the official UCI repository.
