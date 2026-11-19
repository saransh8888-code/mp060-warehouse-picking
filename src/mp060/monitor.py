import sys
import joblib
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from mp060.audit import audit_data
from mp060.clean import clean_dataframe, build_preprocessor
from mp060.train import get_file_sha256

def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.where(exp_counts == 0, 1e-6, exp_counts / len(expected))
    act_pct = np.where(act_counts == 0, 1e-6, act_counts / len(actual))
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

def run_monitoring_and_retrain(best_v1_run_id: str, params_path: str = "params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    raw_train = pd.read_csv(params["data"]["train_path"])
    later_df = pd.read_csv(params["data"]["later_path"])
    holdout_df = pd.read_csv(params["data"]["holdout_path"])
    clean_train = clean_dataframe(raw_train, is_train=True)

    num_cols = params["model"]["num_features"]
    cat_cols = params["model"]["cat_features"]
    features = num_cols + cat_cols
    target = params["model"]["target"]

    psi_scores = {col: round(compute_psi(clean_train[col], later_df[col], bins=10), 4) for col in num_cols}
    most_drifted = max(psi_scores, key=psi_scores.get)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    reg_name = params["mlflow"]["registered_model_name"]
    v1_model = mlflow.sklearn.load_model(f"models:/{reg_name}/1")

    later_preds_v1 = v1_model.predict(later_df[features])
    mae_by_group = {}
    for shift_val in ["Morning", "Evening", "Night"]:
        mask = later_df["shift"] == shift_val
        mae_by_group[shift_val] = round(float(mean_absolute_error(later_df.loc[mask, target], later_preds_v1[mask])), 4)
    worst_group = max(mae_by_group, key=mae_by_group.get)

    later_sorted = later_df.sort_values("date").reset_index(drop=True)
    split_idx = int(0.7 * len(later_sorted))
    later_70 = later_sorted.iloc[:split_idx]
    later_30 = later_sorted.iloc[split_idx:]

    mae_v1_check = round(float(mean_absolute_error(later_30[target], v1_model.predict(later_30[features]))), 4)

    combined_train = pd.concat([clean_train[features + [target]], later_70[features + [target]]], ignore_index=True)
    data_sha256 = get_file_sha256(params["data"]["train_path"])

    with mlflow.start_run(run_name="v2_retrained") as run_v2:
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_type", "GradientBoosting_v2")
        v2_model = Pipeline([
            ("preprocessor", build_preprocessor(num_cols, cat_cols)),
            ("regressor", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)),
        ])
        v2_model.fit(combined_train[features], combined_train[target])
        mae_v2_check = round(float(mean_absolute_error(later_30[target], v2_model.predict(later_30[features]))), 4)
        mlflow.log_metric("mae_check", mae_v2_check)
        mlflow.sklearn.log_model(v2_model, artifact_path="model", serialization_format="cloudpickle")
        v2_run_id = run_v2.info.run_id

    mlflow.register_model(model_uri=f"runs:/{v2_run_id}/model", name=reg_name)

    improvement = (mae_v1_check - mae_v2_check) / mae_v1_check
    decision = "promote" if improvement >= params["model"]["promotion_threshold"] else "keep"
    client = MlflowClient()
    champion_version = 2 if decision == "promote" else 1
    client.set_registered_model_alias(name=reg_name, alias="champion", version=str(champion_version))

    champion_model = mlflow.sklearn.load_model(f"models:/{reg_name}@champion")
    joblib.dump(champion_model, params["model"]["model_path"])

    holdout_clean = clean_dataframe(holdout_df, is_train=False)
    holdout_preds = champion_model.predict(holdout_clean[features])
    sub = pd.DataFrame({"record_id": holdout_df["record_id"], "prediction": np.round(holdout_preds, 2)})
    sub.to_csv(params["data"]["predictions_path"], index=False)

    findings = {
        "group_id": "G256",
        "problem_id": "MP060",
        "data_issues": audit_data(raw_train, later_df, holdout_df),
        "tracking": {
            "experiment_name": params["mlflow"]["experiment_name"],
            "best_run_id": best_v1_run_id,
            "registered_model_name": reg_name,
            "champion_version": champion_version,
        },
        "monitoring": {
            "psi": psi_scores,
            "most_drifted_feature": most_drifted,
            "mae_by_group": mae_by_group,
            "worst_served_group": worst_group,
            "mae_v1_on_check": mae_v1_check,
            "mae_v2_on_check": mae_v2_check,
            "decision": decision,
        },
    }
    with open(params["data"]["findings_path"], "w") as f:
        yaml.safe_dump(findings, f, sort_keys=False)

    print(f"Done! Decision={decision}, v1={mae_v1_check}, v2={mae_v2_check}")

if __name__ == "__main__":
    run_monitoring_and_retrain(sys.argv[1])
