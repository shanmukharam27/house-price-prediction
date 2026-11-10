import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

model_name = "Ames_Housing_Production_Model"

try:
    model = client.get_registered_model(model_name)
    print("MODEL:", model.name)
    print("VERSIONS:")

    versions = client.search_model_versions(f"name='{model_name}'")
    for version in versions:
        print(
            f"version={version.version} "
            f"stage={version.current_stage} "
            f"run_id={version.run_id}"
        )
except Exception as e:
    print(f"Error checking model: {e}")
