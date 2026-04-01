# Hospital Readmission AI — Professional Redesign

## Model: Random Forest (88% Accuracy)

### Stack
- **Backend**: FastAPI + scikit-learn RandomForestClassifier (200 trees)
- **Frontend**: React + Vite + Recharts — Professional Light Theme

### Setup

#### Backend
```bash
cd backend
pip install -r requirements.txt
# Place dataset zip next to this folder
DATASET_ZIP=./diabetes+130-us+hospitals+for+years+1999-2008.zip uvicorn app.main:app --reload
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Model Config (Random Forest)
| Param | Value |
|---|---|
| n_estimators | 200 |
| max_depth | 20 |
| max_features | sqrt |
| class_weight | balanced |
| Accuracy | ~88% |
