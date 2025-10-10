import os, json, joblib, datetime as dt        # Tools: read env vars, JSON, saved model, and time
from typing import Dict, Any                    # For saying “a dict with any values”
from fastapi import FastAPI                     # Web server to make an API
from pydantic import BaseModel                  # Check/parse incoming JSON
import pandas as pd                             # Work with table data
from sqlalchemy import create_engine, text      # Talk to Postgres database

# load model
MODEL = joblib.load("/Users/binhvo/PyCharmMiscProject/ai-ml-portfolio/churn-survival/src/models/artifacts/model.joblib")   # Load the trained model from disk
COLUMNS = json.load(open("/Users/binhvo/PyCharmMiscProject/ai-ml-portfolio/churn-survival/src/models/artifacts/columns.json"))  # Load the expected column names
app = FastAPI(title="Churn API")                # Create the API app

# postgres URL từ env
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://postgres:postgres@db:5432/postgres")
# ^ Get DB link from env; if missing, use this default
engine = create_engine(PG_URL, pool_pre_ping=True)   # Open a connection pool to Postgres

# tạo bảng log nếu chưa có
with engine.begin() as conn:                   # Open a safe DB session (auto-commit/rollback)
    conn.execute(text("""                      # Run SQL to make the log table
    CREATE TABLE IF NOT EXISTS churn_logs (
        id SERIAL PRIMARY KEY,                 # Row id (auto number)
        ts TIMESTAMP,                          # When we made the prediction
        payload JSONB,                         # The input we received (as JSON)
        prob FLOAT,                            # Predicted probability
        pred INT                               # Predicted label 0/1
    )"""))

class ChurnPayload(BaseModel):
    # chấp nhận mọi field, validate tối giản
    __root__: Dict[str, Any]                   # Accept any keys in the JSON

@app.post("/predict")                          # Define a POST endpoint at /predict
def predict(payload: ChurnPayload):            # Function that runs when /predict is called
    data = payload.__root__                    # Get the raw dict from the body

    # cast về DataFrame 1 hàng
    df = pd.DataFrame([data])                  # Make a 1-row table from the JSON

    # đảm bảo đủ cột (thiếu -> NaN)
    for c in COLUMNS["all"]:                   # For each expected column name
        if c not in df.columns: df[c] = None   # If missing, add it with empty value
    df = df[COLUMNS["all"]]                    # Reorder columns to the exact expected order

    prob = float(MODEL.predict_proba(df)[:,1][0])  # Get “churn chance” as a float
    pred = int(prob >= 0.5)                        # Turn chance into 0/1 by 0.5 rule

    # log
    with engine.begin() as conn:               # Open a DB session
        conn.execute(                          # Save a log row to the database
            text("INSERT INTO churn_logs (ts, payload, prob, pred) VALUES (:ts, :payload, :prob, :pred)"),
            {"ts": dt.datetime.utcnow(),       # Current time (UTC)
             "payload": json.dumps(data),      # Save the original input
             "prob": prob,                     # Save probability
             "pred": pred}                     # Save label
        )

    return {"prob": prob, "pred": pred}        # Send result back to the caller
