"""
main_pickle.py
--------------
Lightweight FastAPI app for Render.
Loads a pre-trained model_bundle.pkl — NO training happens here.

Environment variables:
    MODEL_BUNDLE_PATH   path to the pickle file (default: ./model_bundle.pkl)

Run locally:
    uvicorn main_pickle:app --reload

Deploy on Render:
    Start command:  uvicorn main_pickle:app --host 0.0.0.0 --port $PORT
"""

import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_BUNDLE_PATH = os.getenv("MODEL_BUNDLE_PATH", "./model_bundle.pkl")

app = FastAPI(title="Hospital Readmission AI API — Pickle Inference", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state — populated once at startup
STATE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pydantic schema (same as before)
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    race: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    admission_type_id: Optional[int] = None
    discharge_disposition_id: Optional[int] = None
    admission_source_id: Optional[int] = None
    time_in_hospital: Optional[int] = None
    num_lab_procedures: Optional[int] = None
    num_procedures: Optional[int] = None
    num_medications: Optional[int] = None
    number_outpatient: Optional[int] = None
    number_emergency: Optional[int] = None
    number_inpatient: Optional[int] = None
    diag_1: Optional[str] = None
    diag_2: Optional[str] = None
    diag_3: Optional[str] = None
    number_diagnoses: Optional[int] = None
    max_glu_serum: Optional[str] = None
    A1Cresult: Optional[str] = None
    metformin: Optional[str] = None
    repaglinide: Optional[str] = None
    nateglinide: Optional[str] = None
    chlorpropamide: Optional[str] = None
    glimepiride: Optional[str] = None
    acetohexamide: Optional[str] = None
    glipizide: Optional[str] = None
    glyburide: Optional[str] = None
    tolbutamide: Optional[str] = None
    pioglitazone: Optional[str] = None
    rosiglitazone: Optional[str] = None
    acarbose: Optional[str] = None
    miglitol: Optional[str] = None
    troglitazone: Optional[str] = None
    tolazamide: Optional[str] = None
    examide: Optional[str] = None
    citoglipton: Optional[str] = None
    insulin: Optional[str] = None
    glyburide_metformin: Optional[str] = Field(default=None, alias="glyburide-metformin")
    glipizide_metformin: Optional[str] = Field(default=None, alias="glipizide-metformin")
    glimepiride_pioglitazone: Optional[str] = Field(default=None, alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: Optional[str] = Field(default=None, alias="metformin-rosiglitazone")
    metformin_pioglitazone: Optional[str] = Field(default=None, alias="metformin-pioglitazone")
    change: Optional[str] = None
    diabetesMed: Optional[str] = None

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "time_in_hospital": "Time in Hospital (days)",
    "num_lab_procedures": "Number of Lab Procedures",
    "num_procedures": "Number of Procedures",
    "num_medications": "Number of Medications",
    "number_outpatient": "Outpatient Visits",
    "number_emergency": "Emergency Visits",
    "number_inpatient": "Inpatient Visits",
    "number_diagnoses": "Number of Diagnoses",
    "admission_type_id": "Admission Type",
    "discharge_disposition_id": "Discharge Disposition",
    "admission_source_id": "Admission Source",
    "max_glu_serum": "Max Glucose Serum",
    "A1Cresult": "HbA1c Result",
    "metformin": "Metformin",
    "insulin": "Insulin",
    "change": "Medication Change",
    "diabetesMed": "On Diabetes Medication",
}


def humanize_feature(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for prefix, label in [
        ("diag_1_", "Primary Diagnosis: "),
        ("diag_2_", "Secondary Diagnosis: "),
        ("diag_3_", "Additional Diagnosis: "),
        ("age_", "Age Group: "),
        ("race_", "Race: "),
        ("gender_", "Gender: "),
        ("admission_type_id_", "Admission Type: "),
        ("discharge_disposition_id_", "Discharge Disposition: "),
        ("admission_source_id_", "Admission Source: "),
    ]:
        if name.startswith(prefix):
            return label + name[len(prefix):]
    return name.replace("_", " ").title()


def explain_prediction(input_df: pd.DataFrame) -> List[Dict[str, Any]]:
    model = STATE["model"]
    transformed = model.named_steps["preprocessor"].transform(input_df)
    row = (
        transformed[0].toarray().ravel()
        if hasattr(transformed[0], "toarray")
        else np.asarray(transformed[0]).flatten()
    )
    importances = model.named_steps["classifier"].feature_importances_
    df_exp = pd.DataFrame({
        "feature": STATE["feature_names"],
        "importance": importances,
        "encoded_value": row,
    })
    df_exp["contribution"] = df_exp["importance"] * (df_exp["encoded_value"] != 0).astype(float)
    top3 = df_exp[df_exp["contribution"] > 0].sort_values("contribution", ascending=False).head(3)
    return [
        {
            "feature": r["feature"],
            "label": humanize_feature(r["feature"]),
            "value": float(r["encoded_value"]),
            "contribution": float(r["contribution"]),
            "importance": float(r["importance"]),
        }
        for _, r in top3.iterrows()
    ]


def align_input(payload: PredictionRequest) -> pd.DataFrame:
    """Turn request payload into a DataFrame with the same columns the model was trained on."""
    row = payload.model_dump(by_alias=True, exclude_none=False)
    input_df = pd.DataFrame([row])
    # Add any missing columns (model expects them; fill with NaN)
    for col in STATE["X_columns"]:
        if col not in input_df.columns:
            input_df[col] = np.nan
    return input_df[STATE["X_columns"]]


# ---------------------------------------------------------------------------
# Startup — load pickle ONCE
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    global STATE
    if not os.path.exists(MODEL_BUNDLE_PATH):
        raise RuntimeError(
            f"Model bundle not found at '{MODEL_BUNDLE_PATH}'. "
            "Run train_and_save.py locally and upload the resulting pickle."
        )
    print(f"Loading model bundle from {MODEL_BUNDLE_PATH} ...")
    with open(MODEL_BUNDLE_PATH, "rb") as f:
        STATE = pickle.load(f)
    print("Model loaded. Ready to serve.")


# ---------------------------------------------------------------------------
# Routes — identical contract to the original API
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Random Forest (pickle)",
        "n_rows_dataset": STATE.get("n_rows_dataset"),
        "features": len(STATE["feature_names"]),
    }


@app.get("/metrics")
def metrics():
    return STATE["metrics"]


@app.get("/features/top")
def top_features(limit: int = 10):
    return STATE["feature_importance"].head(limit).to_dict(orient="records")


@app.get("/patients/examples")
def patient_examples(limit: int = 10):
    X_sample  = STATE["X_test_sample"]
    y_sample  = STATE["y_test_sample"]
    probs     = STATE["probs_sample"]
    threshold = float(STATE["metrics"]["threshold"])

    limit = max(1, min(limit, len(X_sample)))
    rows = []
    for idx in range(limit):
        raw = X_sample.iloc[idx].to_dict()
        rows.append({
            "id": idx,
            "actual": int(y_sample.iloc[idx]),
            "predicted_probability": float(probs[idx]),
            "predicted_class": int(probs[idx] >= threshold),
            "age": raw.get("age"),
            "gender": raw.get("gender"),
            "time_in_hospital": raw.get("time_in_hospital"),
            "num_medications": raw.get("num_medications"),
            "number_inpatient": raw.get("number_inpatient"),
            "diag_1": raw.get("diag_1"),
        })
    return rows


@app.post("/predict")
def predict(payload: PredictionRequest):
    input_df  = align_input(payload)
    model     = STATE["model"]
    prob      = float(model.predict_proba(input_df)[0, 1])
    threshold = float(STATE["metrics"]["threshold"])
    pred      = int(prob >= threshold)
    return {
        "predicted_probability": prob,
        "predicted_class": pred,
        "threshold": threshold,
        "top_3_risk_drivers": explain_prediction(input_df),
    }


@app.get("/explain/test/{sample_index}")
def explain_test_sample(sample_index: int):
    X_sample  = STATE["X_test_sample"]
    y_sample  = STATE["y_test_sample"]
    probs     = STATE["probs_sample"]
    threshold = float(STATE["metrics"]["threshold"])

    if sample_index < 0 or sample_index >= len(X_sample):
        raise HTTPException(status_code=404, detail=f"sample_index must be 0–{len(X_sample)-1}")

    row  = X_sample.iloc[[sample_index]]
    prob = float(STATE["model"].predict_proba(row)[0, 1])
    pred = int(prob >= threshold)

    return {
        "sample_index": sample_index,
        "actual_class": int(y_sample.iloc[sample_index]),
        "predicted_class": pred,
        "predicted_probability": prob,
        "threshold": threshold,
        "raw_input": row.iloc[0].replace({np.nan: None}).to_dict(),
        "top_3_risk_drivers": explain_prediction(row),
    }


@app.get("/dashboard/summary")
def dashboard_summary():
    return {
        "metrics":          STATE["metrics"],
        "top_features":     STATE["feature_importance"].head(8).to_dict(orient="records"),
        "patient_examples": patient_examples(8),
    }
