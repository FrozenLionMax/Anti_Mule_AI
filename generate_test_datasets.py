import pandas as pd
import numpy as np
import os

def main():
    print("Loading original dataset...")
    df = pd.read_csv("DataSet.csv")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Separate anomalies and normal accounts
    mules = df[df['F3924'] == 1].copy()
    normal = df[df['F3924'] == 0].copy()
    
    # Identify numeric features for manipulation
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['F3924', 'account_id']]
    
    os.makedirs("test_data", exist_ok=True)
    
    # --- 1. EASY DATASET ---
    print("Generating Easy Dataset...")
    easy_normal = normal.sample(n=950, random_state=42)
    easy_mules = mules.sample(n=min(50, len(mules)), replace=True, random_state=42).copy()
    
    # Amplify features artificially to make them obvious
    top_features = numeric_cols[:50]
    for col in top_features:
        easy_mules[col] = easy_mules[col] * np.random.uniform(5, 10, size=len(easy_mules))
        
    easy_df = pd.concat([easy_normal, easy_mules]).sample(frac=1, random_state=42)
    easy_df['account_id'] = [f"EASY_ACC_{i:04d}" for i in range(len(easy_df))]
    easy_df.to_csv("test_data/test_dataset_easy.csv", index=False)
    
    # --- 2. HARD DATASET ---
    print("Generating Hard Dataset...")
    hard_normal = normal.sample(n=970, random_state=100)
    hard_mules = mules.sample(n=min(30, len(mules)), replace=True, random_state=100).copy()
    
    hard_df = pd.concat([hard_normal, hard_mules]).sample(frac=1, random_state=100)
    hard_df['account_id'] = [f"HARD_ACC_{i:04d}" for i in range(len(hard_df))]
    hard_df.to_csv("test_data/test_dataset_hard.csv", index=False)
    
    # --- 3. EXTREME DATASET ---
    print("Generating Extreme Dataset...")
    extreme_normal = normal.sample(n=980, random_state=200)
    extreme_mules = mules.sample(n=min(20, len(mules)), replace=True, random_state=200).copy()
    
    # Force mules to have the exact mean of the normal accounts
    normal_means = normal[numeric_cols].mean()
    for col in numeric_cols:
        extreme_mules[col] = normal_means[col] + np.random.normal(0, 0.01, size=len(extreme_mules))
        
    extreme_df = pd.concat([extreme_normal, extreme_mules]).sample(frac=1, random_state=200)
    extreme_df['account_id'] = [f"EXTREME_ACC_{i:04d}" for i in range(len(extreme_df))]
    extreme_df.to_csv("test_data/test_dataset_extreme.csv", index=False)
    
    print(f"\n--- Summary ---")
    print(f"Easy Dataset:    {len(easy_df)} accounts, {len(easy_mules)} mules.")
    print(f"Hard Dataset:    {len(hard_df)} accounts, {len(hard_mules)} mules.")
    print(f"Extreme Dataset: {len(extreme_df)} accounts, {len(extreme_mules)} mules.")
    print("Datasets saved to test_data/ folder successfully!")

if __name__ == "__main__":
    main()
