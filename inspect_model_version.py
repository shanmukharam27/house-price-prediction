import sys
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

model_name = "Ames_Housing_Production_Model"
version = sys.argv[1] if len(sys.argv) > 1 else "1"

try:
    mv = client.get_model_version(model_name, version)
    print("MODEL NAME:", mv.name)
    print("VERSION:", mv.version)
    print("STAGE:", mv.current_stage)
    print("RUN ID:", mv.run_id)
    print("SOURCE:", mv.source)
    print("STATUS:", mv.status)
    print("CREATION TIMESTAMP:", mv.creation_timestamp)
except Exception as e:
    print(f"Error fetching version {version} of model '{model_name}': {e}")
