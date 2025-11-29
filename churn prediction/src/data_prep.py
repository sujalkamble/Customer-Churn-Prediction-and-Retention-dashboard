"""
Reads the raw Excel file, cleans common issues in the Telco dataset, and writes a cleaned CSV to outputs/cleaned_telco_churn.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils import load_raw, OUTPUT_DIR

OUT = OUTPUT_DIR

def main():
    df = load_raw()
    print('Loaded rows:', df.shape[0])

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Clean TotalCharges: sometimes spaces
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing reasonably
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            mask = df['TotalCharges'].isna()
            df.loc[mask, 'TotalCharges'] = (df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']).fillna(0)

    # Standardize churn column
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    # Some feature engineering: tenure bucket
    if 'tenure' in df.columns:
        bins = [0, 6, 12, 24, 48, 72, 1000]
        labels = ['0-6','7-12','13-24','25-48','49-72','73+']
        df['tenure_bucket'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)

    out_path = OUT / 'cleaned_telco_churn.csv'
    df.to_csv(out_path, index=False)
    print('Saved cleaned dataset to', out_path)

if __name__ == '__main__':
    main()
