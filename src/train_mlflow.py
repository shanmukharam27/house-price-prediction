import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score
)

# Set SQLite tracking URI for reliable local tracking & Model Registry support
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def train_and_track(run_name="Ridge_Baseline", model_type="Ridge", params=None):
    if params is None:
        if model_type == "Ridge":
            params = {"alpha": 10.0, "random_state": 42}
        elif model_type == "RandomForest":
            params = {"n_estimators": 200, "max_depth": 15, "random_state": 42, "n_jobs": -1}
        elif model_type == "XGBoost":
            params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.7, "random_state": 42}

    print(f"\n--- Starting MLflow Run: {run_name} ({model_type}) ---")

    # 1. Load Processed Data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')

    # Set Experiment
    mlflow.set_experiment("House_Price_Prediction")

    with mlflow.start_run(run_name=run_name):
        # 2. Log Parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", model_type)

        # 3. Train Model
        if model_type == "Ridge":
            model = Ridge(**params)
        elif model_type == "RandomForest":
            model = RandomForestRegressor(**params)
        elif model_type == "XGBoost":
            model = XGBRegressor(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(X_train, y_train)

        # 4. Evaluate Predictions
        y_pred_log = model.predict(X_test)
        y_test_dollars = np.expm1(y_test)
        y_pred_dollars = np.expm1(y_pred_log)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_log))
        mae = mean_absolute_error(y_test, y_pred_log)
        r2 = r2_score(y_test, y_pred_log)
        rmse_dollars = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
        mae_dollars = mean_absolute_error(y_test_dollars, y_pred_dollars)

        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "rmse_dollars": float(rmse_dollars),
            "mae_dollars": float(mae_dollars)
        }

        # 5. Log Metrics to MLflow
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: RMSE = {rmse:.4f} | R2 = {r2:.4f} | Dollar RMSE = ${rmse_dollars:,.2f}")

        # 6. Generate and Log Diagnostic Plots (1:1 Parity with churnpredict + Regression plots)
        os.makedirs("artifacts", exist_ok=True)

        # 6.1 Confusion Matrix & ROC Curve (High-Value Property Classification: Price > Median)
        median_threshold = np.median(y_train)
        y_test_binary = (y_test > median_threshold).astype(int)
        y_pred_binary = (y_pred_log > median_threshold).astype(int)

        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test_binary, y_pred_binary, ax=ax_cm, cmap="Blues",
                                                display_labels=["Standard", "High Value"])
        ax_cm.set_title(f"Confusion Matrix - {run_name}")
        cm_path = "artifacts/confusion_matrix.png"
        fig_cm.savefig(cm_path, bbox_inches="tight", dpi=150)
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        # Normalized score as proxy probability
        score_norm = (y_pred_log - y_pred_log.min()) / (y_pred_log.max() - y_pred_log.min() + 1e-8)
        RocCurveDisplay.from_predictions(y_test_binary, score_norm, ax=ax_roc)
        ax_roc.set_title(f"ROC Curve - {run_name}")
        roc_path = "artifacts/roc_curve.png"
        fig_roc.savefig(roc_path, bbox_inches="tight", dpi=150)
        plt.close(fig_roc)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # 6.2 Actual vs Predicted
        fig_avp, ax_avp = plt.subplots(figsize=(7, 6))
        ax_avp.scatter(y_test_dollars, y_pred_dollars, alpha=0.5, color='royalblue', edgecolors='k', s=40)
        min_val = min(y_test_dollars.min(), y_pred_dollars.min())
        max_val = max(y_test_dollars.max(), y_pred_dollars.max())
        ax_avp.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax_avp.set_title(f"Actual vs Predicted - {run_name}", fontsize=12)
        ax_avp.set_xlabel("Actual SalePrice ($)")
        ax_avp.set_ylabel("Predicted SalePrice ($)")
        ax_avp.legend()
        avp_path = "artifacts/actual_vs_predicted.png"
        fig_avp.savefig(avp_path, bbox_inches="tight", dpi=150)
        plt.close(fig_avp)
        mlflow.log_artifact(avp_path, artifact_path="plots")

        # 6.3 Residuals Distribution
        residuals = y_test - y_pred_log
        fig_res, ax_res = plt.subplots(figsize=(7, 5))
        sns.histplot(residuals, kde=True, ax=ax_res, color='forestgreen')
        ax_res.axvline(x=0, color='red', linestyle='--')
        ax_res.set_title(f"Residuals Distribution - {run_name}", fontsize=12)
        ax_res.set_xlabel("Residual (Actual Log - Predicted Log)")
        res_path = "artifacts/residuals_distribution.png"
        fig_res.savefig(res_path, bbox_inches="tight", dpi=150)
        plt.close(fig_res)
        mlflow.log_artifact(res_path, artifact_path="plots")

        # 6.4 Feature Importance / Coefficients
        fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-15:]
            ax_imp.barh(range(len(top_idx)), importances[top_idx], color='teal')
            ax_imp.set_yticks(range(len(top_idx)))
            ax_imp.set_yticklabels([f"Feature {i}" for i in top_idx])
            ax_imp.set_title(f"Top 15 Feature Importances - {run_name}")
        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_)
            top_idx = np.argsort(coefs)[-15:]
            ax_imp.barh(range(len(top_idx)), coefs[top_idx], color='darkorange')
            ax_imp.set_yticks(range(len(top_idx)))
            ax_imp.set_yticklabels([f"Feature {i}" for i in top_idx])
            ax_imp.set_title(f"Top 15 Absolute Coefficients - {run_name}")
        imp_path = "artifacts/feature_importance.png"
        fig_imp.savefig(imp_path, bbox_inches="tight", dpi=150)
        plt.close(fig_imp)
        mlflow.log_artifact(imp_path, artifact_path="plots")

        # 7. Log Preprocessing Metadata for Lineage
        if os.path.exists('data/processed/dataset_metadata.json'):
            mlflow.log_artifact('data/processed/dataset_metadata.json', artifact_path="metadata")

        # 8. Log Model Artifact
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree", "xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"]
            )
        except Exception:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle"
            )

        # Save local backup
        os.makedirs("models", exist_ok=True)
        if model_type == "Ridge":
            joblib.dump(model, 'models/ridge_baseline.pkl')
        elif model_type == "RandomForest":
            joblib.dump(model, 'models/random_forest_model.pkl')
        elif model_type == "XGBoost":
            joblib.dump(model, 'models/xgboost_model.pkl')

        print(f"Run '{run_name}' successfully tracked!")

if __name__ == "__main__":
    # Baseline Run: Random Forest
    train_and_track(run_name="RandomForest", model_type="RandomForest")

    # Tuned Tree Model Run: XGBoost
    train_and_track(run_name="XGBoost_Tuned", model_type="XGBoost")
