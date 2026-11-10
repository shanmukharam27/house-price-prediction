import sys
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Get the latest run ID if none provided
exp = client.get_experiment_by_name("House_Price_Prediction")
if exp:
    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
    run_id = sys.argv[1] if len(sys.argv) > 1 else (runs[0].info.run_id if runs else None)
else:
    run_id = sys.argv[1] if len(sys.argv) > 1 else None

if not run_id:
    print("No runs found in experiment 'House_Price_Prediction'")
    sys.exit(0)

run = client.get_run(run_id)

print("RUN ID:", run.info.run_id)
print("STATUS:", run.info.status)
print("ARTIFACT URI:", run.info.artifact_uri)
print("\nMETRICS:")
for k, v in run.data.metrics.items():
    print(f"  {k}: {v}")

print("\nARTIFACTS:")
artifacts = client.list_artifacts(run_id)
for artifact in artifacts:
    print(f"  {artifact.path} | is_dir={artifact.is_dir}")
