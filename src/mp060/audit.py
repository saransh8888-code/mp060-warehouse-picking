import pandas as pd
import yaml

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def audit_data(train_df: pd.DataFrame, later_df: pd.DataFrame, holdout_df: pd.DataFrame) -> dict:
    issues = {}

    missing_in_holdout = [c for c in train_df.columns if c not in holdout_df.columns and c != "pick_seconds"]
    leaking_col = None
    for c in missing_in_holdout:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            if abs(train_df[c].corr(train_df["pick_seconds"])) > 0.95:
                leaking_col = c
    issues["leaking_column"] = {"present": leaking_col is not None, "column": leaking_col}

    unit_issue = {"present": False, "column": None, "group_value": None, "factor": None}
    for col in ["items_in_order", "aisle_distance_m", "picker_exp_months", "shelf_height_m", "shift_hour", "order_weight_kg"]:
        means = train_df.groupby("shift")[col].mean()
        ratio = means.max() / means.min()
        if ratio > 50:
            unit_issue = {
                "present": True,
                "column": col,
                "group_value": str(means.idxmax()),
                "factor": 100 if round(ratio) == 100 else int(round(ratio)),
            }
    issues["unit_mixup"] = unit_issue

    sentinel_issue = {"present": False, "column": None, "value": None, "count": None}
    for col in ["items_in_order", "aisle_distance_m", "picker_exp_months", "shelf_height_m", "shift_hour", "order_weight_kg"]:
        vc = train_df[col].value_counts()
        if 999.0 in vc.index and vc[999.0] > 10:
            sentinel_issue = {"present": True, "column": col, "value": 999, "count": int(vc[999.0])}
    issues["sentinel_value"] = sentinel_issue

    dup_count = int(train_df.duplicated().sum())
    issues["duplicate_rows"] = {"present": dup_count > 0, "count": dup_count if dup_count > 0 else None}

    sorted_df = train_df.sort_values("date").reset_index(drop=True)
    stuck_issue = {"present": False, "column": None, "start_date": None, "end_date": None}
    for col in ["items_in_order", "aisle_distance_m", "picker_exp_months", "shelf_height_m", "shift_hour", "order_weight_kg"]:
        runs = (sorted_df[col] != sorted_df[col].shift()).cumsum()
        summary = sorted_df.groupby(runs).agg(count=(col, "size"), start=("date", "min"), end=("date", "max"))
        longest = summary.sort_values("count", ascending=False).iloc[0]
        if longest["count"] >= 20:
            stuck_issue = {
                "present": True,
                "column": col,
                "start_date": str(longest["start"]),
                "end_date": str(longest["end"]),
            }
    issues["stuck_sensor"] = stuck_issue

    q99 = train_df["pick_seconds"].quantile(0.99)
    typos = int((train_df["pick_seconds"] > q99 * 5).sum())
    issues["target_typos"] = {"present": typos > 0, "count": typos if typos > 0 else None}

    variant_issue = {"present": False, "column": None, "true_levels": None}
    for col in ["zone_type", "shift"]:
        if set(train_df[col].unique()) != set(later_df[col].unique()):
            variant_issue = {"present": True, "column": col, "true_levels": int(later_df[col].nunique())}
    issues["spelling_variants"] = variant_issue

    return issues

if __name__ == "__main__":
    params = load_params()
    train = pd.read_csv(params["data"]["train_path"])
    later = pd.read_csv(params["data"]["later_path"])
    holdout = pd.read_csv(params["data"]["holdout_path"])
    print(yaml.dump(audit_data(train, later, holdout), sort_keys=False))
