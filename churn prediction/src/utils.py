import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

def load_raw(path=None):
    if path is None:
        path = DATA_DIR / 'Telco_customer_churn.xlsx'
    return pd.read_excel(path)
