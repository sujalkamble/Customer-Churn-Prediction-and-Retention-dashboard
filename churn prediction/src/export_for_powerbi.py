"""
Score the cleaned dataset with the saved XGBoost model and export:
1) predictions_with_proba.csv      → raw predictions
2) final_churn_powerbi.csv         → Power BI-ready file
3) feature_importances.csv         → Top features affecting churn
"""

from pathlib import Path
import pandas as pd
import joblib
from src.utils import OUTPUT_DIR

OUT = OUTPUT_DIR
MODEL_PATH = OUT / 'churn_pipeline.joblib'

def main():
    # -------------------- Load cleaned data --------------------
    cleaned = Path(__file__).resolve().parents[1] / 'outputs' / 'cleaned_telco_churn.csv'
    if not cleaned.exists():
        raise FileNotFoundError('cleaned_telco_churn.csv not found in outputs/. Run data_prep.py first')
    df = pd.read_csv(cleaned)

    # -------------------- Load trained XGBoost pipeline --------------------
    pipe = joblib.load(MODEL_PATH)

    # -------------------- Predict probabilities --------------------
    X = df.drop(columns=['Churn Value'])
    proba = pipe.predict_proba(X)[:, 1]
    predicted_churn = (proba >= 0.5).astype(int)

    # -------------------- Raw predictions export --------------------
    raw_pred_df = df.copy()
    raw_pred_df['churn_probability'] = proba
    raw_pred_df['predicted_churn'] = predicted_churn
    raw_pred_df.to_csv(OUT / 'predictions_with_proba.csv', index=False)
    print(f"✔ Saved main prediction file → predictions_with_proba.csv")

    # -------------------- Power BI export --------------------
    # Include all original columns + predictions
    powerbi_df = df.copy()
    powerbi_df['churn_probability'] = proba
    powerbi_df['predicted_churn'] = predicted_churn
    powerbi_df['actual_churn'] = df['Churn Value']
    powerbi_df.to_csv(OUT / 'final_churn_powerbi.csv', index=False)
    print(f"🔥 Power BI file generated → final_churn_powerbi.csv")

    # -------------------- Feature importance export --------------------
    if hasattr(pipe.named_steps['clf'], 'feature_importances_'):
        importances = pipe.named_steps['clf'].feature_importances_

        # Numeric features
        numeric_cols = pipe.named_steps['preprocessor'].named_transformers_['num'].feature_names_in_

        # One-hot encoded categorical features
        ohe = pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
        categorical_cols = ohe.get_feature_names_out()

        # Combine features
        final_features = list(numeric_cols) + list(categorical_cols)

        fi_df = pd.DataFrame({
            "Feature": final_features,
            "Importance": importances
        }).sort_values('Importance', ascending=False)

        fi_df.to_csv(OUT / 'feature_importances.csv', index=False)
        print(f"📈 Feature Importance exported → feature_importances.csv")
    else:
        print("⚠ Model does not support feature_importances_. Can't export rankings.")

    print("\n🎉 ALL FILES GENERATED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
