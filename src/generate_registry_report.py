import os
import json
import mlflow
from mlflow.tracking import MlflowClient

# Set SQLite tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")
    
    client = MlflowClient()
    model_name = "House_Price_Production_Model"
    
    try:
        # Search for the model specifically in Production
        production_models = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in production_models if mv.current_stage == "Production"), None)
        
        if not champion:
            print("[ERROR] No model found in Production stage!")
            return
            
        # Get run details to extract metrics and parameters
        run = client.get_run(champion.run_id)
        
        # Build deployment-ready artifact dictionary
        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source
            },
            "performance_metrics": {
                "rmse": run.data.metrics.get("rmse"),
                "mae": run.data.metrics.get("mae"),
                "r2_score": run.data.metrics.get("r2_score"),
                "rmse_dollars": run.data.metrics.get("rmse_dollars"),
                "mae_dollars": run.data.metrics.get("mae_dollars")
            },
            "hyperparameters": run.data.params,
            "target_variable": "SalePrice",
            "target_transformation": "log1p",
            "preprocessing_dependency": "models/preprocessor.pkl"
        }
        
        # Save report
        os.makedirs("artifacts", exist_ok=True)
        report_path = "artifacts/production_model_report.json"
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        print(f"[SUCCESS] Report successfully generated for Version {champion.version}")
        print(f"[INFO] Saved to: {report_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")

if __name__ == "__main__":
    generate_registry_report()
