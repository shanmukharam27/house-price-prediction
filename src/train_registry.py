import os
import numpy as np
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")
    
    # 1. Set up MLflow
    mlflow.set_experiment("House_Price_Prediction")
    
    # 2. Load Processed Data
    try:
        X_train = np.load('data/processed/X_train_final.npy')
        X_test = np.load('data/processed/X_test_final.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_test = np.load('data/processed/y_test.npy')
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run the Lab 5 pipeline first.")
        return
    
    # 3. Define Model Parameters (Best performing XGBoost architecture)
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.7,
        "random_state": 42
    }
    
    with mlflow.start_run(run_name="XGBoost_Registry_Candidate") as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", "XGBoost")
        
        # 4. Train Model
        print("[INFO] Training XGBoost model...")
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # 5. Evaluate
        print("[INFO] Evaluating model...")
        y_pred = model.predict(X_test)
        
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2
        }
        
        mlflow.log_metrics(metrics)
        
        # 6. LOG PREPROCESSOR PIPELINE (Lineage)
        if os.path.exists("models/preprocessor.pkl"):
            print("[INFO] Attaching preprocessing pipeline to model artifacts...")
            mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessing_pipeline")
            
        # 7. LOG AND REGISTER THE MODEL
        print("[INFO] Pushing model to MLflow Registry...")
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name="House_Price_Production_Model",
                skops_trusted_types=["sklearn.tree._tree.Tree", "xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"]
            )
        except Exception:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name="House_Price_Production_Model",
                serialization_format="cloudpickle"
            )
            
        print(f"[SUCCESS] Metrics logged: RMSE = {rmse:.4f} | R² = {r2:.4f}")
        print(f"[SUCCESS] Model successfully registered under name: 'House_Price_Production_Model'")

if __name__ == "__main__":
    train_and_register_model()
