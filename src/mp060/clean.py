import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    out = df.copy()
    if "pick_time_from_scanner" in out.columns:
        out = out.drop(columns=["pick_time_from_scanner"])

    if is_train:
        if "shift" in out.columns and "shelf_height_m" in out.columns:
            out.loc[out["shift"] == "Morning", "shelf_height_m"] = (
                out.loc[out["shift"] == "Morning", "shelf_height_m"] / 100.0
            )
        if "order_weight_kg" in out.columns:
            out.loc[out["order_weight_kg"] == 999.0, "order_weight_kg"] = np.nan

        if "date" in out.columns and "shift_hour" in out.columns:
            mask = (out["date"] >= "2026-03-13") & (out["date"] <= "2026-04-04")
            out.loc[mask, "shift_hour"] = np.nan

    return out

def build_preprocessor(num_features: list, cat_features: list) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])
