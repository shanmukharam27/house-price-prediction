import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_evaluation():
    print("Starting Model Evaluation...")
    
    # 1. Load test data and trained baseline model
    X_test_final = np.load('data/processed/X_test_final.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    model_path = 'models/random_forest_baseline.pkl'
    if not os.path.exists(model_path):
        model_path = 'models/ridge_baseline.pkl'
        
    model = joblib.load(model_path)
    
    # 2. Make predictions in log-scale
    y_pred_log = model.predict(X_test_final)
    
    # Convert predictions and ground truth back to dollar scale
    y_test_dollars = np.expm1(y_test)
    y_pred_dollars = np.expm1(y_pred_log)
    
    # 3. Calculate metrics
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    mae_log = mean_absolute_error(y_test, y_pred_log)
    r2 = r2_score(y_test, y_pred_log)
    rmse_dollars = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
    mae_dollars = mean_absolute_error(y_test_dollars, y_pred_dollars)
    
    print("\n--- Model Evaluation Report ---")
    print(f"Model Evaluated : {model_path}")
    print(f"Log RMSE        : {rmse_log:.4f}")
    print(f"Log MAE         : {mae_log:.4f}")
    print(f"R² Score        : {r2:.4f}")
    print(f"Dollar RMSE     : ${rmse_dollars:,.2f}")
    print(f"Dollar MAE      : ${mae_dollars:,.2f}")
    print("-------------------------------\n")
    
    # 4. Error Analysis Export (Parity with churnpredict outputs)
    os.makedirs('outputs', exist_ok=True)
    
    residuals = y_test - y_pred_log
    dollar_errors = y_test_dollars - y_pred_dollars
    pct_errors = np.abs(dollar_errors) / y_test_dollars * 100
    
    error_df = pd.DataFrame({
        'Actual_Log_Price': y_test,
        'Predicted_Log_Price': y_pred_log,
        'Actual_SalePrice': y_test_dollars.round(2),
        'Predicted_SalePrice': y_pred_dollars.round(2),
        'Dollar_Error': dollar_errors.round(2),
        'Absolute_Pct_Error': pct_errors.round(2),
        'Residual_Log': residuals.round(4)
    })
    
    # High residual cases
    high_residuals = error_df.sort_values(by='Absolute_Pct_Error', ascending=False)
    high_residuals.to_csv('outputs/high_residual_errors.csv', index=False)
    
    # Overpredictions (predicted > actual) and Underpredictions (actual > predicted)
    overpredicted = error_df[error_df['Dollar_Error'] < 0].sort_values(by='Dollar_Error')
    underpredicted = error_df[error_df['Dollar_Error'] > 0].sort_values(by='Dollar_Error', ascending=False)
    
    overpredicted.to_csv('outputs/overpredicted_cases.csv', index=False)
    underpredicted.to_csv('outputs/underpredicted_cases.csv', index=False)
    
    # Direct 1:1 filename parity with churnpredict
    # False positives = model over-estimated value (predicted high when actual was low)
    # False negatives = model under-estimated value (predicted low when actual was high)
    overpredicted.to_csv('outputs/false_positives.csv', index=False)
    underpredicted.to_csv('outputs/false_negatives.csv', index=False)
    
    print("Evaluation complete and error analysis files saved!")
    print("  -> Saved outputs/false_positives.csv & outputs/overpredicted_cases.csv")
    print("  -> Saved outputs/false_negatives.csv & outputs/underpredicted_cases.csv")
    print("  -> Saved outputs/high_residual_errors.csv")
    
    # 5. Generate Kaggle Submission if test dataset is processed
    if os.path.exists('data/processed/X_submission_final.npy') and os.path.exists('data/processed/test_ids.npy'):
        X_sub = np.load('data/processed/X_submission_final.npy')
        test_ids = np.load('data/processed/test_ids.npy')
        sub_preds_log = model.predict(X_sub)
        sub_preds = np.expm1(sub_preds_log)
        
        submission = pd.DataFrame({
            'Id': test_ids,
            'SalePrice': sub_preds
        })
        submission_path = 'outputs/submission.csv'
        submission.to_csv(submission_path, index=False)
        print(f"Kaggle submission file generated at: {submission_path}")

if __name__ == "__main__":
    run_evaluation()
