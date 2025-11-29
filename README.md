# Customer-Churn-Prediction-and-Retention-dashboard
Machine learning model to predict telecom customer churn using Python &amp; SQL with Power BI dashboard insights, EDA, and predictive analysis for retention improvement.

Customer Churn Prediction           #Project Root Folder
│
├── data/
│   ├── raw/                        # Original datasets
│   │    └── Telco_customer_churn.xlsv
│   ├── processed/                  # Cleaned / preprocessed data
│   │    └── churn_data_clean.csv
│
├── src/                            # Source code / scripts
│   ├── data_prep.py                # Cleaning, encoding, scaling
│   ├── train_model.py              # Train ML model (XGBoost/RandomForest)
│   ├── predict.py                  # Model evaluation & metrics
│   ├── utils.py
│   ├── export_for_powerbi.py       # Export predictions + probabilities
│
├── cinfig.yaml                     # store configuration settings for your project
│
├── outputs/
│   ├── churn_pipeline.joblib       # serialized machine learning pipeline
│   ├── cleaned_telco_churn.csv     # For train ML model
│   ├── feature_importances.csv     # used for analyzing & visualizing top features
│   ├── predictions_with_proba.csv  # Prediction results
│   └── final_churn_powerbi.csv     # Data for Power BI dashboard
│
├── dashboard/
│   └── retention_dashboard.pbix     # Retention dashboard
│   └── retention_dashboard.png      # Retention dashboard screenshot
│
├── requirements.txt                 # Python libraries & versions
├── README.md                        # Project documentation
└── Churn prediction documentation   # full project document
