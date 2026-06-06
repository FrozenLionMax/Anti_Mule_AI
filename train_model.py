import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import joblib

def calculate_top_k_metrics(y_true, y_prob, k):
    """Calculate Precision and Recall at Top-K."""
    if len(y_true) < k:
        k = len(y_true)
    
    # Sort indices by predicted probability descending
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    
    # Get true labels for the top K instances
    top_k_true = y_true.iloc[top_k_indices].values
    
    # In the top K selected, how many are actually anomalies (precision)
    precision_at_k = np.sum(top_k_true) / k
    
    # In the top K selected, what fraction of all anomalies were caught (recall)
    total_anomalies = np.sum(y_true)
    recall_at_k = np.sum(top_k_true) / total_anomalies if total_anomalies > 0 else 0.0
    
    return precision_at_k, recall_at_k

def main():
    parser = argparse.ArgumentParser(description="Train AntiMuleAI Model")
    parser.add_argument("--leakage-safe", action="store_true", help="Run in leakage-safe mode (drops leaky features)")
    args = parser.parse_args()

    print("=== AntiMuleAI Model Training ===")
    
    # 1. Load Data
    data_path = "DataSet.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Assume 'account_id' is present. If not, create a dummy one.
    if 'account_id' not in df.columns:
        print("Creating dummy 'account_id' column...")
        df['account_id'] = [f"ACC_{i:06d}" for i in range(len(df))]
        
    # Set target
    target_col = 'F3924'
    if target_col not in df.columns:
        print(f"Error: Target column {target_col} not found!")
        return

    # 2. Drop useless columns
    print("Cleaning columns...")
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Drop all-null and constant columns
    initial_cols = len(df.columns)
    df = df.dropna(axis=1, how='all')
    
    # To handle object vs numeric unique counting efficiently
    nunique = df.nunique()
    cols_to_drop = nunique[nunique <= 1].index
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {initial_cols - len(df.columns)} useless columns (all-null or constant).")

    # Leakage-safe mode
    if args.leakage_safe:
        leaky_cols = ['F3912', 'F2230'] # Examples based on prompt
        drop_leaky = [c for c in leaky_cols if c in df.columns]
        if drop_leaky:
            df = df.drop(columns=drop_leaky)
            print(f"Leakage-safe mode: Dropped leaky columns: {drop_leaky}")

    y = df[target_col]
    
    # Keep account_id for output mapping
    account_ids = df['account_id']
    
    # Drop target and ID from features
    X = df.drop(columns=[target_col, 'account_id'])

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # 4 & 5. Preprocessing Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 7. Stratified Train/Test Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Train size: {len(X_train)} (Anomalies: {y_train.sum()})")
    print(f"Test size: {len(X_test)} (Anomalies: {y_test.sum()})")

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import VotingClassifier

    # 6. Train Models
    # Calculate scale_pos_weight dynamically to handle the severe class imbalance
    neg_count = len(y_train) - y_train.sum()
    pos_count = y_train.sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"\nCalculated scale_pos_weight for Boosting models: {scale_pos_weight:.2f}")

    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=42, n_jobs=-1)
    
    # XGBoost and LightGBM with extreme focus on the minority class
    xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
    lgbm_model = LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42, n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1, verbose=-1)
    
    # Professional Ensemble Architecture
    voting_model = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model), ('lgbm', lgbm_model)],
        voting='soft',
        n_jobs=-1
    )

    models = {
        'RandomForest': rf_model,
        'XGBoost': xgb_model,
        'LightGBM': lgbm_model,
        'Ensemble (Voting)': voting_model
    }

    best_auc = -1
    best_model_name = ""
    best_pipeline = None

    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # 8. Metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC:  {pr_auc:.4f}")
        
        for k in [25, 50, 100]:
            p_at_k, r_at_k = calculate_top_k_metrics(y_test, y_prob, k)
            print(f"Precision@{k}: {p_at_k:.4f} | Recall@{k}: {r_at_k:.4f}")
            
        if pr_auc > best_auc:
            best_auc = pr_auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest model selected: {best_model_name} (PR-AUC: {best_auc:.4f})")
    
    # 9. Save Best Model
    model_path = os.path.join("models", "muleguard_model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"Model saved to {model_path}")

    # Generate full predictions for risk_scores.csv
    print("Generating risk scores for all accounts...")
    all_probs = best_pipeline.predict_proba(X)[:, 1]
    
    risk_df = pd.DataFrame({
        'account_id': account_ids,
        'risk_score': np.round(all_probs * 100, 2),
        'actual_label': y
    })
    
    # Thresholding for predicted label
    threshold = 80.0
    risk_df['predicted_label'] = (risk_df['risk_score'] >= threshold).astype(int)
    
    # Risk Bands
    def assign_band(score):
        if score >= 80: return 'High'
        elif score >= 40: return 'Medium'
        else: return 'Low'
        
    risk_df['risk_band'] = risk_df['risk_score'].apply(assign_band)
    
    # Sort by risk score
    risk_df = risk_df.sort_values(by='risk_score', ascending=False)
    
    output_path = os.path.join("outputs", "risk_scores.csv")
    risk_df.to_csv(output_path, index=False)
    print(f"Risk scores saved to {output_path}")
    print("Training complete!")

if __name__ == "__main__":
    main()
