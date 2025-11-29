"""
Load the saved pipeline and score a new single customer (example dictionary format)
"""
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / 'outputs' / 'churn_pipeline.joblib'


def predict_single(customer_dict):
    pipe = joblib.load(MODEL_PATH)
    df = pd.DataFrame([customer_dict])
    proba = pipe.predict_proba(df)[:,1][0]
    return proba

if __name__ == '__main__':
    sample = {
        'gender':'Female',
        'SeniorCitizen':0,
        'Partner':'Yes',
        'Dependents':'No',
        'tenure':1,
        'PhoneService':'Yes',
        'MultipleLines':'No',
        'InternetService':'DSL',
        'OnlineSecurity':'No',
        'OnlineBackup':'Yes',
        'DeviceProtection':'No',
        'TechSupport':'No',
        'StreamingTV':'No',
        'StreamingMovies':'No',
        'Contract':'Month-to-month',
        'PaperlessBilling':'Yes',
        'PaymentMethod':'Electronic check',
        'MonthlyCharges':29.85,
        'TotalCharges':29.85
    }
    print('Predicted churn probability:', predict_single(sample))
