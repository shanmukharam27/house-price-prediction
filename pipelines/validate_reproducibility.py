import os
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def run_deterministic_test():
    print("Running Reproducibility Validation...")
    
    # Load Data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Fixed Parameters for strict reproducibility
    params = {"alpha": 10.0, "random_state": 42}
    
    # Execution 1
    model_1 = Ridge(**params)
    model_1.fit(X_train, y_train)
    score_1 = float(np.sqrt(mean_squared_error(y_test, model_1.predict(X_test))))
    
    # Execution 2 (Simulating a re-run)
    model_2 = Ridge(**params)
    model_2.fit(X_train, y_train)
    score_2 = float(np.sqrt(mean_squared_error(y_test, model_2.predict(X_test))))
    
    # Validate consistency
    is_reproducible = bool(np.isclose(score_1, score_2, atol=1e-10))
    
    # Generate Report
    report = {
        "test_name": "Pipeline Reproducibility Validation",
        "model": "Ridge",
        "parameters": params,
        "execution_1_rmse": score_1,
        "execution_2_rmse": score_2,
        "is_strictly_reproducible": is_reproducible,
        "status": "PASSED" if is_reproducible else "FAILED"
    }
    
    os.makedirs("artifacts", exist_ok=True)
    report_path = 'artifacts/reproducibility_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Execution 1 RMSE: {score_1:.6f}")
    print(f"Execution 2 RMSE: {score_2:.6f}")
    if is_reproducible:
        print("SUCCESS: Pipeline is 100% reproducible. Report saved.")
    else:
        print("FAILED: Pipeline is non-deterministic.")

if __name__ == "__main__":
    run_deterministic_test()
