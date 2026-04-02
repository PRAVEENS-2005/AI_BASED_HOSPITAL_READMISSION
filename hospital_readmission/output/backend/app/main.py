import os
import zipfile
import urllib.request
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

# ===== CONFIG =====
DATASET_ZIP = os.getenv(
    "DATASET_ZIP",
    "./diabetes+130-us+hospitals+for+years+1999-2008.zip"
)

DATASET_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

# ===== APP =====
app = FastAPI(title="Hospital Readmission AI API — Random Forest", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict[str, Any] = {}

# ===== MODELS =====
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


# ===== DATA LOADING =====
def load_dataset(path: str) -> pd.DataFrame:
    # Debug info
    print("CWD:", os.getcwd())
    print("FILES:", os.listdir("."))

    if not os.path.exists(path):
        print(f"Dataset not found at {path}, downloading...")

        os.makedirs("/tmp", exist_ok=True)
        download_path = "/tmp/dataset.zip"

        urllib.request.urlretrieve(DATASET_URL, download_path)
        path = download_path

    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("diabetic_data.csv") as f:
            df = pd.read_csv(f)

    return df


# ===== PREPROCESS =====
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

    rf_classifier = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
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

    return {
        "df": df,
        "X": X,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
    }


# ===== STARTUP =====
@app.on_event("startup")
def startup_event():
    global STATE
    print("=== STARTING APP ===")
    STATE = prepare_data(load_dataset(DATASET_ZIP))
    print("=== MODEL READY ===")


# ===== ROUTES =====
@app.get("/")
def root():
    return {"message": "API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}
