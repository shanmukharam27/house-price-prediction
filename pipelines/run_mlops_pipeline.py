import subprocess
import sys

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Full End-to-End MLOps Pipeline")
    print("[INFO] Project: Ames House Price Prediction")
    print("[INFO] =========================================")
    
    stages = [
        ("Data Preprocessing & Feature Engineering", "src/preprocess_pipeline.py"),
        ("Data Schema Validation", "src/validate_data.py"),
        ("Model Training & MLflow Experiment Tracking", "src/train_mlflow.py"),
        ("Model Evaluation & Diagnostic Reporting", "src/evaluate.py"),
        ("Model Registry & Production Quality Gate", "src/generate_registry_report.py"),
        ("Strict Deterministic Reproducibility Test", "pipelines/validate_reproducibility.py"),
    ]
    
    for stage_name, script_path in stages:
        print(f"\n[INFO] >>> Running Stage: {stage_name} ({script_path})")
        res = subprocess.run([sys.executable, script_path])
        if res.returncode != 0:
            print(f"[ERROR] Stage '{stage_name}' failed with return code {res.returncode}.")
            sys.exit(res.returncode)
            
    print("\n[SUCCESS] ========================================================")
    print("[SUCCESS] All End-to-End MLOps Pipeline Stages Passed Successfully!")
    print("[SUCCESS] ========================================================")

if __name__ == "__main__":
    execute_pipeline()
