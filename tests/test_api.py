import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from mp060.clean import clean_dataframe

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_predict_valid():
    r = client.post("/predict", json={"items_in_order": 15, "aisle_distance_m": 150.0, "picker_exp_months": 30.0, "shelf_height_m": 1.4, "shift_hour": 10, "order_weight_kg": 12.5, "zone_type": "ambient", "shift": "Morning"})
    assert r.status_code == 200 and "prediction" in r.json()

def test_predict_rules():
    base = {"items_in_order": 15, "aisle_distance_m": 150.0, "picker_exp_months": 30.0, "shelf_height_m": 1.4, "shift_hour": 10, "order_weight_kg": 12.5, "zone_type": "ambient", "shift": "Morning"}
    assert client.post("/predict", json={**base, "picker_exp_months": 105.0}).status_code == 422
    assert client.post("/predict", json={**base, "aisle_distance_m": 4.5}).status_code == 422

def test_clean():
    df = pd.DataFrame({"date": ["2026-03-15"], "shift": ["Morning"], "shelf_height_m": [150.0], "order_weight_kg": [999.0], "shift_hour": [13], "pick_time_from_scanner": [200]})
    c = clean_dataframe(df, is_train=True)
    assert "pick_time_from_scanner" not in c.columns and c.loc[0, "shelf_height_m"] == 1.5
