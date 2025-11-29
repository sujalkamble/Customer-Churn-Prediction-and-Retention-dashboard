"""
Train XGBoost churn model + generate files for Power BI:
✔ churn_pipeline_xgb.joblib
✔ feature_importances.csv
✔ predictions_with_proba.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from src.utils import OUTPUT_DIR

OUT = OUTPUT_DIR

# config
TARGET = 'Churn Value'
MODEL_PATH = OUT / 'churn_pipeline.joblib'
FEATURE_IMP_PATH = OUT / 'feature_importances.csv'
PRED_EXPORT_PATH = OUT / 'predictions_with_proba.csv'

def main():
    # 1) Load data
    try:
        df = pd.read_csv(OUT / 'cleaned_telco_churn.csv')
    except:
        df = pd.read_csv(Path(__file__).resolve().parents[1] / 'outputs' / 'cleaned_telco_churn.csv')

    if TARGET not in df.columns:
        raise ValueError("❌ Target column not found: 'Churn Value'")

    # 2) Split X/Y
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # 3) Identify column types
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    # 4) Preprocessor
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ])

    # 5) XGBoost Model
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=42
    )

    pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])

    # 6) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("🚀 Training XGBoost Model...")
    pipe.fit(X_train, y_train)

    print("📊 Evaluating...")
    y_proba = pipe.predict_proba(X_test)[:,1]
    y_pred = pipe.predict(X_test)
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    # 7) Save model
    joblib.dump(pipe, MODEL_PATH)
    print(f"✔ Model saved to: {MODEL_PATH}")

    # 8) 🔥 Export Feature Importances
    ohe = pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
    ohe_features = ohe.get_feature_names_out(categorical_cols)
    final_features = list(numeric_cols) + list(ohe_features)
    importances = pipe.named_steps['clf'].feature_importances_

    fi_df = pd.DataFrame({
        "Feature": final_features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    fi_df.to_csv(FEATURE_IMP_PATH, index=False)
    print(f"📄 Feature importances exported → {FEATURE_IMP_PATH}")

    # 9) 🔥 Export Predictions
    result_df = X_test.copy()
    result_df["Actual Churn"] = y_test
    result_df["Predicted Probability"] = y_proba
    result_df["Predicted Churn"] = (y_proba >= 0.5).astype(int)
    result_df.to_csv(PRED_EXPORT_PATH, index=False)
    print(f"📄 Predictions exported → {PRED_EXPORT_PATH}")
    print("🎉 All files ready for Power BI dashboard.")

if __name__ == "__main__":
    main()
