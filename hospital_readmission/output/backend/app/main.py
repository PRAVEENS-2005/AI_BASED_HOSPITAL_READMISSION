import os
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_ZIP = os.getenv("DATASET_ZIP", "./diabetes+130-us+hospitals+for+years+1999-2008.zip")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

app = FastAPI(title="Hospital Readmission AI API — Random Forest", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict[str, Any] = {}


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


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset zip not found at {path}. Set DATASET_ZIP env variable."
        )
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("diabetic_data.csv") as f:
            df = pd.read_csv(f)
    return df


def prepare_data(df: pd.DataFrame):
    df = df.replace("?", np.nan)
    df = df.drop(columns=["weight", "payer_code", "medical_specialty"], errors="ignore")
    df["target"] = (df["readmitted"] == "<30").astype(int)

    X = df.drop(columns=["readmitted", "target", "encounter_id", "patient_nbr"], errors="ignore")
    y = df["target"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    # Random Forest — targets 88%+ accuracy
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf_classifier),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importance from Random Forest
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(cat_feature_names)

    importances = clf.feature_importances_
    feature_importance = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances,
        "abs_coefficient": importances,  # for frontend compatibility
    })
    feature_importance = feature_importance.sort_values("importance", ascending=False)

    actual_accuracy = float(accuracy_score(y_test, y_pred))

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "feature_names": all_feature_names,
        "feature_importance": feature_importance,
        "metrics": {
            "accuracy": actual_accuracy,
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "threshold": PREDICTION_THRESHOLD,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "support": int(len(y_test)),
            "model_type": "Random Forest",
            "n_estimators": 200,
            "n_features": len(all_feature_names),
            "training_samples": int(len(X_train)),
        },
    }


def to_row_dict(payload: PredictionRequest) -> Dict[str, Any]:
    return payload.model_dump(by_alias=True, exclude_none=False)


def humanize_feature(feature_name: str) -> str:
    if feature_name.startswith("diag_1_"):
        return f"Primary diagnosis: {feature_name.replace('diag_1_', '')}"
    if feature_name.startswith("diag_2_"):
        return f"Secondary diagnosis: {feature_name.replace('diag_2_', '')}"
    if feature_name.startswith("diag_3_"):
        return f"Additional diagnosis: {feature_name.replace('diag_3_', '')}"
    if feature_name.startswith("age_"):
        return f"Age group {feature_name.replace('age_', '')}"
    if feature_name.startswith("race_"):
        return f"Race: {feature_name.replace('race_', '')}"
    if feature_name.startswith("gender_"):
        return f"Gender: {feature_name.replace('gender_', '')}"
    return feature_name.replace("_", " ").title()


def explain_prediction(input_df: pd.DataFrame) -> List[Dict[str, Any]]:
    model: Pipeline = STATE["model"]
    transformed = model.named_steps["preprocessor"].transform(input_df)
    row = np.asarray(transformed[0]).flatten()

    clf = model.named_steps["classifier"]
    importances = clf.feature_importances_

    df_exp = pd.DataFrame({
        "feature": STATE["feature_names"],
        "importance": importances,
        "encoded_value": row,
    })
    # Use importance weighted by non-zero features
    df_exp["contribution"] = df_exp["importance"] * (df_exp["encoded_value"] != 0).astype(float)
    positive = df_exp[df_exp["contribution"] > 0].sort_values("contribution", ascending=False)

    out = []
    for _, r in positive.head(3).iterrows():
        out.append({
            "feature": r["feature"],
            "label": humanize_feature(r["feature"]),
            "value": float(r["encoded_value"]),
            "contribution": float(r["contribution"]),
            "importance": float(r["importance"]),
        })
    return out


@app.on_event("startup")
def startup_event():
    global STATE
    STATE = prepare_data(load_dataset(DATASET_ZIP))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Random Forest",
        "dataset_loaded": True,
        "rows": int(len(STATE["df"])),
        "features": int(len(STATE["feature_names"])),
    }


@app.get("/metrics")
def metrics():
    return STATE["metrics"]


@app.get("/features/top")
def top_features(limit: int = 10):
    return STATE["feature_importance"].head(limit).to_dict(orient="records")


@app.get("/patients/examples")
def patient_examples(limit: int = 10):
    X_test = STATE["X_test"].reset_index(drop=True)
    y_test = STATE["y_test"].reset_index(drop=True)
    model: Pipeline = STATE["model"]
    probs = model.predict_proba(X_test)[:, 1]
    limit = max(1, min(limit, len(X_test)))
    rows = []
    for idx in range(limit):
        raw = X_test.iloc[idx].to_dict()
        rows.append({
            "id": idx,
            "actual": int(y_test.iloc[idx]),
            "predicted_probability": float(probs[idx]),
            "predicted_class": int(probs[idx] >= PREDICTION_THRESHOLD),
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
    model: Pipeline = STATE["model"]
    input_df = pd.DataFrame([to_row_dict(payload)])
    missing_cols = [c for c in STATE["X"].columns if c not in input_df.columns]
    for col in missing_cols:
        input_df[col] = np.nan
    input_df = input_df[STATE["X"].columns]

    prob = float(model.predict_proba(input_df)[0, 1])
    pred = int(prob >= PREDICTION_THRESHOLD)
    return {
        "predicted_probability": prob,
        "predicted_class": pred,
        "threshold": PREDICTION_THRESHOLD,
        "top_3_risk_drivers": explain_prediction(input_df),
    }


@app.get("/explain/test/{sample_index}")
def explain_test_sample(sample_index: int):
    X_test = STATE["X_test"].reset_index(drop=True)
    y_test = STATE["y_test"].reset_index(drop=True)
    if sample_index < 0 or sample_index >= len(X_test):
        raise HTTPException(status_code=404, detail="sample index out of range")
    row = X_test.iloc[[sample_index]]
    model: Pipeline = STATE["model"]
    prob = float(model.predict_proba(row)[0, 1])
    pred = int(prob >= PREDICTION_THRESHOLD)
    return {
        "sample_index": sample_index,
        "actual_class": int(y_test.iloc[sample_index]),
        "predicted_class": pred,
        "predicted_probability": prob,
        "raw_input": row.iloc[0].replace({np.nan: None}).to_dict(),
        "top_3_risk_drivers": explain_prediction(row),
    }


@app.get("/dashboard/summary")
def dashboard_summary():
    return {
        "metrics": STATE["metrics"],
        "top_features": STATE["feature_importance"].head(8).to_dict(orient="records"),
        "patient_examples": patient_examples(8),
    }
