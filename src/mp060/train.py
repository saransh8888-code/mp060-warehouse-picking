import hashlib
import joblib
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from mp060.clean import clean_dataframe, build_preprocessor

def get_file_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def run_training(params_path: str = "params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    train_path = params["data"]["train_path"]
    data_sha256 = get_file_sha256(train_path)

    raw_df = pd.read_csv(train_path)
    clean_df = clean_dataframe(raw_df, is_train=True)

    num_cols = params["model"]["num_features"]
    cat_cols = params["model"]["cat_features"]
    features = num_cols + cat_cols
    target = params["model"]["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        clean_df[features], clean_df[target], test_size=0.2, random_state=params["model"]["random_state"]
    )

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    candidates = [
        ("Ridge", Ridge(alpha=1.0), {"model_type": "Ridge", "alpha": 1.0}),
        ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=42), {"model_type": "RandomForest", "n_estimators": 200}),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42), {"model_type": "GradientBoosting", "n_estimators": 150}),
    ]

    best_mae = float("inf")
    best_run_id = None

    for name, estimator, hyperparams in candidates:
        with mlflow.start_run(run_name=f"v1_{name}") as run:
            mlflow.log_param("data_sha256", data_sha256)
            for k, v in hyperparams.items():
                mlflow.log_param(k, v)

            pipe = Pipeline([
                ("preprocessor", build_preprocessor(num_cols, cat_cols)),
                ("regressor", estimator),
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)
            mae = float(mean_absolute_error(y_val, preds))
            mlflow.log_metric("mae_val", mae)

            pipe.fit(clean_df[features], clean_df[target])
            mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="cloudpickle")

            print(f"{name}: val MAE = {mae:.4f} | run_id = {run.info.run_id}")
            if mae < best_mae:
                best_mae = mae
                best_run_id = run.info.run_id

    reg_name = params["mlflow"]["registered_model_name"]
    mv = mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name=reg_name)

    client = MlflowClient()
    client.set_registered_model_alias(name=reg_name, alias="champion", version=mv.version)

    champion_model = mlflow.sklearn.load_model(f"models:/{reg_name}@champion")
    joblib.dump(champion_model, params["model"]["model_path"])
    print(f"\nBEST_RUN_ID={best_run_id}")
    return best_run_id

if __name__ == "__main__":
    run_training()
