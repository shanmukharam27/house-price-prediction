import pandas as pd
import numpy as np
from src.validate_data import validate_schema

def simulate_corrupted_data(df):
    """Simulates real-world data corruption scenarios across housing dataset columns."""
    print("[INFO] Injecting simulated faults into Ames housing data...")
    corrupted_df = df.copy()
    
    # 1. Invalid numerical values (negative LotArea)
    if 'LotArea' in corrupted_df.columns:
        corrupted_df.loc[0, 'LotArea'] = -500
    
    # 2. Out of bounds categorical/numerical (OverallQual > 10)
    if 'OverallQual' in corrupted_df.columns:
        corrupted_df.loc[1, 'OverallQual'] = 15
        
    # 3. Invalid YearBuilt (outside range)
    if 'YearBuilt' in corrupted_df.columns:
        corrupted_df.loc[2, 'YearBuilt'] = 1750
        
    # 4. Negative SalePrice
    if 'SalePrice' in corrupted_df.columns:
        corrupted_df.loc[3, 'SalePrice'] = -10000
    
    # 5. Schema inconsistency (unexpected extra column)
    corrupted_df['unexpected_column'] = 9999
    
    # 6. Missing required attribute (drop GrLivArea)
    if 'GrLivArea' in corrupted_df.columns:
        corrupted_df = corrupted_df.drop(columns=['GrLivArea'])
    
    return corrupted_df

if __name__ == "__main__":
    data_path = 'data/raw/train.csv'
    
    try:
        raw_df = pd.read_csv(data_path)
        print(f"[INFO] Loaded raw training data ({len(raw_df)} rows)")
        
        # 1. Validate clean dataset first
        print("\n--- Validating Clean Dataset ---")
        validate_schema(raw_df)
        
        # 2. Inject corruptions
        print("\n--- Validating Corrupted Dataset ---")
        corrupted = simulate_corrupted_data(raw_df)
        validate_schema(corrupted, output_report_name="schema_corruption_failures.csv")
        
    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
