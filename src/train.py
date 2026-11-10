import os
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

def run_training():
    print("Starting Model Training...")
    
    # 1. Load processed training data
    X_train_final = np.load('data/processed/X_train_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    
    os.makedirs('models', exist_ok=True)
    
    # 2. Train Ridge baseline
    print("Training Ridge regression baseline...")
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_train_final, y_train)
    joblib.dump(ridge_model, 'models/ridge_baseline.pkl')
    
    # 3. Train Random Forest baseline
    print("Training Random Forest baseline...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_final, y_train)
    joblib.dump(rf_model, 'models/random_forest_baseline.pkl')
    
    print("Model training complete and baseline models saved to disk!")

if __name__ == "__main__":
    run_training()
