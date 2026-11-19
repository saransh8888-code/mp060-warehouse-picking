import json, logging, os, sys, time, joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

class JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {"timestamp": self.formatTime(record, self.datefmt), "level": record.levelname, "message": record.getMessage()}
        if hasattr(record, "extra_data"): d.update(record.extra_data)
        return json.dumps(d)

logger = logging.getLogger("mp060_api")
logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(JsonFormatter())
logger.handlers = [h]

app = FastAPI(title="MP060 Warehouse Picking Time API")
model = joblib.load(os.getenv("MODEL_PATH", "models/model.joblib")) if os.path.exists(os.getenv("MODEL_PATH", "models/model.joblib")) else None

class PredictRequest(BaseModel):
    items_in_order: int
    aisle_distance_m: float = Field(..., ge=5)
    picker_exp_months: float = Field(..., le=100)
    shelf_height_m: float
    shift_hour: int
    order_weight_kg: float
    zone_type: str
    shift: str

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    pred = float(model.predict(pd.DataFrame([req.model_dump()]))[0]) if model else 0.0
    logger.info("prediction_served", extra={"extra_data": {"prediction": round(pred, 2), "latency_ms": round((time.time()-t0)*1000, 2)}})
    return {"prediction": pred}
