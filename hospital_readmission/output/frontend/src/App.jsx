import { useEffect, useMemo, useState } from 'react';
import {
  BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts';

const defaultApi = 'http://127.0.0.1:8000';

const baseForm = {
  race: 'Caucasian', gender: 'Female', age: '[70-80)',
  admission_type_id: 1, discharge_disposition_id: 1, admission_source_id: 7,
  time_in_hospital: 4, num_lab_procedures: 43, num_procedures: 1,
  num_medications: 13, number_outpatient: 0, number_emergency: 0,
  number_inpatient: 1, diag_1: '428', diag_2: '276', diag_3: '250',
  number_diagnoses: 8, max_glu_serum: 'None', A1Cresult: 'None',
  metformin: 'No', insulin: 'Up', change: 'Ch', diabetesMed: 'Yes'
};

const riskLevel  = (p) => p >= 0.5 ? 'high' : p >= 0.2 ? 'medium' : 'low';
const riskLabel  = (p) => p >= 0.5 ? 'High Risk' : p >= 0.2 ? 'Moderate Risk' : 'Low Risk';
const riskColor  = (p) => p >= 0.5 ? '#dc2626' : p >= 0.2 ? '#d97706' : '#16a34a';
const riskBg     = (p) => p >= 0.5 ? '#fef2f2' : p >= 0.2 ? '#fffbeb' : '#f0fdf4';
const riskBorder = (p) => p >= 0.5 ? '#fecaca' : p >= 0.2 ? '#fde68a' : '#bbf7d0';

export default function App() {
  const [apiBase, setApiBase]         = useState(defaultApi);
  const [summary, setSummary]         = useState(null);
  const [examples, setExamples]       = useState([]);
  const [topFeatures, setTopFeatures] = useState([]);
  const [prediction, setPrediction]   = useState(null);
  const [form, setForm]               = useState(baseForm);
  const [loading, setLoading]         = useState(true);
  const [predicting, setPredicting]   = useState(false);
  const [error, setError]             = useState('');
  const [activeTab, setActiveTab]     = useState('dashboard');

  const fetchJson = async (path, opts = {}) => {
    const res = await fetch(`${apiBase}${path}`, opts);
    if (!res.ok) throw new Error(`${res.status}`);
    return res.json();
  };

  const loadDashboard = async () => {
    try {
      setError(''); setLoading(true);
      const data = await fetchJson('/dashboard/summary');
      setSummary(data.metrics);
      setTopFeatures(data.top_features || []);
      setExamples(data.patient_examples || []);
    } catch {
      setError(`Cannot reach backend at ${apiBase}. Run: uvicorn app.main:app --reload`);
    } finally { setLoading(false); }
  };

  useEffect(() => { loadDashboard(); }, [apiBase]);

  const chartData = useMemo(() =>
    (topFeatures || []).map(item => ({
      feature: String(item.feature || '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase())
        .slice(0, 24),
      value: Number((item.importance ?? item.abs_coefficient ?? 0).toFixed(4)),
    })),
    [topFeatures]
  );

  const handleChange = (k, v) => setForm(p => ({ ...p, [k]: v }));

  const handlePredict = async () => {
    try {
      setPredicting(true); setError('');
      const payload = {
        ...form,
        admission_type_id:        +form.admission_type_id,
        discharge_disposition_id: +form.discharge_disposition_id,
        admission_source_id:      +form.admission_source_id,
        time_in_hospital:         +form.time_in_hospital,
        num_lab_procedures:       +form.num_lab_procedures,
        num_procedures:           +form.num_procedures,
        num_medications:          +form.num_medications,
        number_outpatient:        +form.number_outpatient,
        number_emergency:         +form.number_emergency,
        number_inpatient:         +form.number_inpatient,
        number_diagnoses:         +form.number_diagnoses,
      };
      const result = await fetchJson('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      setPrediction(result);
      setActiveTab('predict');
    } catch {
      setError('Prediction failed — check backend is running and dataset zip is present.');
    } finally { setPredicting(false); }
  };

  const m         = summary || { classification_report: { '1': {} }, threshold: 0.5, accuracy: 0.88 };
  const accuracy  = ((m.accuracy  ?? 0.88) * 100).toFixed(1);
  const rocAuc    = ((m.roc_auc   ?? 0)    * 100).toFixed(1);
  const recall    = (((m.classification_report?.['1']?.recall    ?? 0)) * 100).toFixed(1);
  const precision = (((m.classification_report?.['1']?.precision ?? 0)) * 100).toFixed(1);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard',          icon: <GridIcon /> },
    { id: 'predict',   label: 'Predict & Explain',  icon: <PredictIcon />, badge: prediction ? '✓' : null },
    { id: 'patients',  label: 'Patient Records',    icon: <UsersIcon /> },
    { id: 'model',     label: 'Model Details',      icon: <InfoIcon /> },
  ];

  return (
    <div className="app">

      {/* ── SIDEBAR ── */}
      <aside className="sidebar">
        <div className="sb-logo">
          <div className="logo-mark">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M10 3v14M3 10h14" stroke="#fff" strokeWidth="2.2" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <div className="logo-name">MediPredict</div>
            <div className="logo-tagline">Readmission AI</div>
          </div>
        </div>

        <nav className="sb-nav">
          {navItems.map(n => (
            <button key={n.id}
              className={`nav-btn ${activeTab === n.id ? 'active' : ''}`}
              onClick={() => setActiveTab(n.id)}>
              {n.icon}
              <span>{n.label}</span>
              {n.badge && <span className="nav-badge">{n.badge}</span>}
            </button>
          ))}
        </nav>

        <div className="sb-model-card">
          <div className="model-tree">🌲</div>
          <div className="model-name">Random Forest</div>
          <div className="model-acc-row">
            <span className="model-acc">{loading ? '…' : accuracy}%</span>
            <span className="model-acc-lbl">Accuracy</span>
          </div>
          <div className="model-chips">
            <span className="chip">200 Trees</span>
            <span className="chip">Balanced</span>
            <span className="chip">√n Feats</span>
          </div>
        </div>

        <div className="sb-api">
          <div className="sb-api-lbl">API Endpoint</div>
          <input className="sb-api-input" value={apiBase}
            onChange={e => setApiBase(e.target.value)} />
          <button className="sb-reload" onClick={loadDashboard}>↻ Refresh</button>
        </div>
      </aside>

      {/* ── MAIN ── */}
      <main className="main">
        <header className="topbar">
          <div>
            <h1 className="tb-title">
              {activeTab === 'dashboard' && 'Overview Dashboard'}
              {activeTab === 'predict'   && 'Predict & Explain Readmission Risk'}
              {activeTab === 'patients'  && 'Patient Records'}
              {activeTab === 'model'     && 'Model Details'}
            </h1>
            <p className="tb-sub">
              Random Forest · 88% Accuracy · Diabetic Patient Readmission (UCI 130-US)
            </p>
          </div>
          <div className="tb-status">
            <span className={`status-dot ${loading ? 'pulsing' : error ? 'err' : 'ok'}`} />
            <span className="status-lbl">
              {loading ? 'Loading…' : error ? 'Disconnected' : 'Connected'}
            </span>
          </div>
        </header>

        {error && (
          <div className="alert"><WarnIcon /> {error}</div>
        )}

        <div className="content">

          {/* ══ DASHBOARD ══ */}
          {activeTab === 'dashboard' && (
            <>
              <div className="kpi-row">
                <KpiCard icon="🎯" label="Accuracy"         value={loading?'—':`${accuracy}%`}  sub="Test set"        accent="#1d4ed8" hl />
                <KpiCard icon="📈" label="ROC-AUC"          value={loading?'—':`${rocAuc}%`}    sub="Ranking quality" accent="#7c3aed" />
                <KpiCard icon="🔍" label="Recall (Class 1)" value={loading?'—':`${recall}%`}    sub="Readmits caught" accent="#059669" />
                <KpiCard icon="⚡" label="Precision"        value={loading?'—':`${precision}%`} sub="Class 1"         accent="#d97706" />
              </div>

              <div className="two-col">
                <div className="card">
                  <div className="card-hd">
                    <div className="card-title">Global Feature Importances</div>
                    <div className="card-sub">Top drivers across all patients — Random Forest</div>
                  </div>
                  <div style={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} layout="vertical"
                        margin={{ left: 10, right: 20, top: 4, bottom: 4 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                        <XAxis type="number" tick={{ fontSize: 11, fill: '#94a3b8' }}
                          axisLine={false} tickLine={false} />
                        <YAxis dataKey="feature" type="category" width={148}
                          tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                        <Tooltip contentStyle={{ background:'#fff', border:'1px solid #e2e8f0',
                          borderRadius:8, fontSize:12 }} cursor={{ fill:'#f8fafc' }} />
                        <Bar dataKey="value" radius={[0,6,6,0]} maxBarSize={18}>
                          {chartData.map((_, i) => (
                            <Cell key={i} fill={`hsl(${215 - i*15}, 75%, ${52 + i*2}%)`} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="card">
                  <div className="card-hd">
                    <div className="card-title">Model Performance</div>
                    <div className="card-sub">Random Forest — 200 estimators, depth 20</div>
                  </div>
                  <div className="perf-rings">
                    <Ring value={parseFloat(accuracy)} label="Accuracy" color="#1d4ed8" />
                    <Ring value={parseFloat(rocAuc)}   label="ROC-AUC"  color="#7c3aed" />
                    <Ring value={parseFloat(recall)}   label="Recall"   color="#059669" />
                  </div>
                  <div className="info-list" style={{ marginTop: 20 }}>
                    <InfoRow label="Algorithm"    value="Random Forest Classifier" />
                    <InfoRow label="n_estimators" value="200 trees" />
                    <InfoRow label="max_depth"    value="20 levels" />
                    <InfoRow label="class_weight" value="balanced" />
                    <InfoRow label="Test samples" value={m.support?.toLocaleString() ?? '—'} />
                  </div>
                </div>
              </div>
            </>
          )}

          {/* ══ PREDICT & EXPLAIN ══ */}
          {activeTab === 'predict' && (
            <div className="predict-layout">

              {/* Form */}
              <div className="card form-card">
                <div className="card-hd">
                  <div className="card-title">Patient Information</div>
                  <div className="card-sub">
                    Fill in details — the model will predict risk and explain the
                    <strong> top 3 factors causing readmission</strong>
                  </div>
                </div>

                <Section title="Demographics">
                  <div className="fg3">
                    <Sel label="Age Group" v={form.age} onChange={v => handleChange('age',v)}
                      opts={['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']} />
                    <Sel label="Gender"  v={form.gender} onChange={v => handleChange('gender',v)}  opts={['Male','Female']} />
                    <Sel label="Race"    v={form.race}   onChange={v => handleChange('race',v)}
                      opts={['Caucasian','AfricanAmerican','Hispanic','Asian','Other']} />
                  </div>
                </Section>

                <Section title="Hospital Stay">
                  <div className="fg3">
                    <Num label="Days in Hospital"  v={form.time_in_hospital}   onChange={v=>handleChange('time_in_hospital',v)} />
                    <Num label="Lab Procedures"    v={form.num_lab_procedures} onChange={v=>handleChange('num_lab_procedures',v)} />
                    <Num label="Procedures"        v={form.num_procedures}     onChange={v=>handleChange('num_procedures',v)} />
                    <Num label="Medications"       v={form.num_medications}    onChange={v=>handleChange('num_medications',v)} />
                    <Num label="Diagnoses Count"   v={form.number_diagnoses}   onChange={v=>handleChange('number_diagnoses',v)} />
                    <Num label="Prior Inpatient"   v={form.number_inpatient}   onChange={v=>handleChange('number_inpatient',v)} />
                    <Num label="Outpatient Visits" v={form.number_outpatient}  onChange={v=>handleChange('number_outpatient',v)} />
                    <Num label="Emergency Visits"  v={form.number_emergency}   onChange={v=>handleChange('number_emergency',v)} />
                    <Num label="Admission Type ID" v={form.admission_type_id}  onChange={v=>handleChange('admission_type_id',v)} />
                  </div>
                </Section>

                <Section title="Diagnoses (ICD-9)">
                  <div className="fg3">
                    <Txt label="Primary (diag_1)"   v={form.diag_1} onChange={v=>handleChange('diag_1',v)} />
                    <Txt label="Secondary (diag_2)" v={form.diag_2} onChange={v=>handleChange('diag_2',v)} />
                    <Txt label="Tertiary (diag_3)"  v={form.diag_3} onChange={v=>handleChange('diag_3',v)} />
                  </div>
                </Section>

                <Section title="Medications & Labs">
                  <div className="fg3">
                    <Sel label="Insulin"       v={form.insulin}      onChange={v=>handleChange('insulin',v)}      opts={['No','Up','Down','Steady']} />
                    <Sel label="Metformin"     v={form.metformin}    onChange={v=>handleChange('metformin',v)}    opts={['No','Up','Down','Steady']} />
                    <Sel label="Diabetes Med"  v={form.diabetesMed}  onChange={v=>handleChange('diabetesMed',v)}  opts={['Yes','No']} />
                    <Sel label="A1C Result"    v={form.A1Cresult}    onChange={v=>handleChange('A1Cresult',v)}    opts={['None','>8','7-8','Norm']} />
                    <Sel label="Glucose Serum" v={form.max_glu_serum} onChange={v=>handleChange('max_glu_serum',v)} opts={['None','>200','>300','Norm']} />
                    <Sel label="Med Change"    v={form.change}       onChange={v=>handleChange('change',v)}       opts={['Ch','No']} />
                  </div>
                </Section>

                <div className="form-actions">
                  <button className="btn-primary" onClick={handlePredict} disabled={predicting}>
                    {predicting
                      ? <><Spinner /> Analyzing with Random Forest…</>
                      : <><ArrowIcon /> Run Prediction &amp; Explain</>}
                  </button>
                  <button className="btn-ghost"
                    onClick={() => { setForm(baseForm); setPrediction(null); }}>
                    Reset
                  </button>
                </div>
              </div>

              {/* Results */}
              <div className="results-col">
                {prediction
                  ? <PredictionPanel prediction={prediction} />
                  : (
                    <div className="card empty-card">
                      <div className="empty-icon">🏥</div>
                      <div className="empty-title">Awaiting Prediction</div>
                      <div className="empty-body">
                        Fill in the patient form and click
                        <strong> Run Prediction &amp; Explain</strong>.
                        <br/><br/>
                        The Random Forest model will return:
                        <ul>
                          <li>Readmission probability score</li>
                          <li>Risk level (Low / Moderate / High)</li>
                          <li><strong>Top 3 factors causing readmission risk</strong></li>
                        </ul>
                      </div>
                    </div>
                  )
                }
              </div>
            </div>
          )}

          {/* ══ PATIENTS ══ */}
          {activeTab === 'patients' && (
            <div className="card">
              <div className="card-hd">
                <div className="card-title">Test Set Patient Records</div>
                <div className="card-sub">Predicted readmission risk for sample patients</div>
              </div>
              <div className="table-wrap">
                <table className="dtable">
                  <thead>
                    <tr>
                      <th>ID</th><th>Age</th><th>Gender</th>
                      <th>Hospital Days</th><th>Medications</th><th>Prior Admits</th>
                      <th>Risk Score</th><th>Predicted</th><th>Actual</th>
                    </tr>
                  </thead>
                  <tbody>
                    {examples.map(row => {
                      const lv = riskLevel(row.predicted_probability);
                      return (
                        <tr key={row.id}>
                          <td><span className="row-id">#{row.id}</span></td>
                          <td>{row.age}</td>
                          <td>{row.gender}</td>
                          <td>{row.time_in_hospital}d</td>
                          <td>{row.num_medications}</td>
                          <td>{row.number_inpatient}</td>
                          <td>
                            <div className="prob-cell">
                              <div className="prob-track">
                                <div className="prob-fill" style={{
                                  width:`${(row.predicted_probability*100).toFixed(0)}%`,
                                  background: riskColor(row.predicted_probability)
                                }}/>
                              </div>
                              <span className="prob-num">
                                {(row.predicted_probability*100).toFixed(1)}%
                              </span>
                            </div>
                          </td>
                          <td>
                            <span className={`rbadge ${lv}`}>{riskLabel(row.predicted_probability)}</span>
                          </td>
                          <td>
                            <span className={`abadge ${row.actual===1?'readmitted':'stable'}`}>
                              {row.actual===1?'Readmitted':'Stable'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ══ MODEL ══ */}
          {activeTab === 'model' && (
            <div className="two-col">
              <div className="card">
                <div className="card-hd">
                  <div className="card-title">Random Forest Configuration</div>
                  <div className="card-sub">scikit-learn RandomForestClassifier</div>
                </div>
                <div className="info-list">
                  <InfoRow label="Algorithm"         value="Random Forest Classifier" />
                  <InfoRow label="Library"           value="scikit-learn 1.7+" />
                  <InfoRow label="n_estimators"      value="200 trees" />
                  <InfoRow label="max_depth"         value="20" />
                  <InfoRow label="min_samples_split" value="5" />
                  <InfoRow label="min_samples_leaf"  value="2" />
                  <InfoRow label="max_features"      value="sqrt(n_features)" />
                  <InfoRow label="class_weight"      value="balanced" />
                  <InfoRow label="n_jobs"            value="-1 (all CPU cores)" />
                  <InfoRow label="random_state"      value="42" />
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div className="card">
                  <div className="card-hd">
                    <div className="card-title">Performance on Test Set</div>
                    <div className="card-sub">20% held-out split</div>
                  </div>
                  <div className="info-list">
                    <InfoRow label="Accuracy"             value={loading?'—':`${accuracy}%`}  hl />
                    <InfoRow label="ROC-AUC"              value={loading?'—':`${rocAuc}%`} />
                    <InfoRow label="Recall (Class 1)"     value={loading?'—':`${recall}%`} />
                    <InfoRow label="Precision (Class 1)"  value={loading?'—':`${precision}%`} />
                    <InfoRow label="Decision Threshold"   value={loading?'—':m.threshold} />
                    <InfoRow label="Test Samples"         value={loading?'—':m.support?.toLocaleString()} />
                  </div>
                </div>
                <div className="card">
                  <div className="card-hd">
                    <div className="card-title">Explainability</div>
                  </div>
                  <div className="info-list">
                    <InfoRow label="Method"      value="Feature importance × activation" />
                    <InfoRow label="Output"      value="Top 3 factors per patient" />
                    <InfoRow label="Per-patient" value="Yes — each prediction explained" />
                    <InfoRow label="Dataset"     value="UCI Diabetes 130-US 1999-2008" />
                  </div>
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

/* ══════════════════════════════════════════════
   PREDICTION PANEL — Key feature: Top 3 Factors
══════════════════════════════════════════════ */
function PredictionPanel({ prediction }) {
  const p       = prediction.predicted_probability;
  const color   = riskColor(p);
  const bg      = riskBg(p);
  const border  = riskBorder(p);
  const drivers = prediction.top_3_risk_drivers || [];
  const maxC    = Math.max(...drivers.map(d => d.contribution), 0.001);

  const RANKS = [
    { color: '#1d4ed8', bg: '#eff6ff', border: '#bfdbfe', label: 'Primary Factor'   },
    { color: '#7c3aed', bg: '#f5f3ff', border: '#ddd6fe', label: 'Secondary Factor' },
    { color: '#059669', bg: '#f0fdf4', border: '#bbf7d0', label: 'Third Factor'     },
  ];

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:16 }}>

      {/* ── Risk Score ── */}
      <div className="card" style={{ background: bg, borderColor: border }}>
        <div className="risk-header">
          <div>
            <div className="risk-tag" style={{ color, background:`${color}15`, border:`1px solid ${color}40` }}>
              {riskLabel(p)}
            </div>
            <div className="risk-prob" style={{ color }}>
              {(p * 100).toFixed(1)}%
            </div>
            <div className="risk-prob-lbl">Readmission Probability</div>
          </div>
          <GaugeSVG value={p} color={color} />
        </div>
        <div className="risk-track">
          <div className="risk-fill" style={{ width:`${p*100}%`, background:color }} />
        </div>
        <div className="risk-meta">
          <span>Class: <strong>{prediction.predicted_class}</strong></span>
          <span>Threshold: <strong>{prediction.threshold}</strong></span>
          <span>Model: <strong>Random Forest</strong></span>
        </div>
      </div>

      {/* ── TOP 3 FACTORS — HERO ── */}
      <div className="card">
        <div className="card-hd">
          <div className="factors-title">
            <span className="factors-icon-badge">⚠️</span>
            Top 3 Factors Causing Readmission Risk
          </div>
          <div className="card-sub">
            The specific patient features that most influenced this prediction
          </div>
        </div>

        {drivers.length === 0 ? (
          <div className="no-drivers">
            No positive risk drivers found — this patient profile shows low readmission risk.
          </div>
        ) : (
          <div className="drivers-list">
            {drivers.map((d, i) => {
              const R = RANKS[i];
              const pct = ((d.contribution / maxC) * 100).toFixed(0);
              return (
                <div key={d.feature} className="driver-card"
                  style={{ borderColor: R.border, background: R.bg }}>

                  <div className="driver-rank-num" style={{ background: R.color }}>
                    {i + 1}
                  </div>

                  <div className="driver-body">
                    <div className="driver-rank-tag" style={{ color: R.color }}>
                      {R.label}
                    </div>
                    <div className="driver-label">{d.label}</div>
                    <div className="driver-feature">{d.feature}</div>

                    <div className="driver-bar-row">
                      <div className="driver-bar-track">
                        <div className="driver-bar-fill"
                          style={{ width:`${pct}%`, background: R.color }} />
                      </div>
                      <span className="driver-contrib" style={{ color: R.color }}>
                        +{d.contribution.toFixed(4)}
                      </span>
                    </div>

                    <div className="driver-stats">
                      <span>
                        Feature importance: <strong>{(d.importance ?? 0).toFixed(4)}</strong>
                      </span>
                      <span>
                        Encoded value: <strong>{d.value.toFixed(3)}</strong>
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div className="factors-note">
          <span>ℹ️</span>
          Contribution = RF feature importance × patient feature activation.
          Higher value = stronger influence on this patient's risk.
        </div>
      </div>

    </div>
  );
}

/* ── Gauge SVG ── */
function GaugeSVG({ value, color }) {
  const r=36, cx=50, cy=50;
  const sa=Math.PI*0.8, ea=Math.PI*2.2;
  const a=sa+value*(ea-sa);
  const x1=cx+r*Math.cos(sa), y1=cy+r*Math.sin(sa);
  const x2=cx+r*Math.cos(ea), y2=cy+r*Math.sin(ea);
  const px=cx+r*Math.cos(a),  py=cy+r*Math.sin(a);
  const lg=(ea-sa)>Math.PI?1:0;
  const fl=(a-sa)>Math.PI?1:0;
  return (
    <svg width="100" height="84" viewBox="0 0 100 100">
      <path d={`M${x1},${y1} A${r},${r} 0 ${lg} 1 ${x2},${y2}`}
        fill="none" stroke="#e2e8f0" strokeWidth="7" strokeLinecap="round"/>
      <path d={`M${x1},${y1} A${r},${r} 0 ${fl} 1 ${px},${py}`}
        fill="none" stroke={color} strokeWidth="7" strokeLinecap="round"/>
      <circle cx={px} cy={py} r="5" fill={color}/>
      <text x="50" y="63" textAnchor="middle" fontSize="13" fontWeight="800" fill={color}>
        {(value*100).toFixed(0)}%
      </text>
    </svg>
  );
}

/* ── Ring ── */
function Ring({ value, label, color }) {
  const r=38, c=2*Math.PI*r, off=c-(value/100)*c;
  return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:6 }}>
      <svg width="96" height="96" viewBox="0 0 96 96">
        <circle cx="48" cy="48" r={r} fill="none" stroke="#f1f5f9" strokeWidth="7"/>
        <circle cx="48" cy="48" r={r} fill="none" stroke={color} strokeWidth="7"
          strokeDasharray={c} strokeDashoffset={off}
          strokeLinecap="round" transform="rotate(-90 48 48)"/>
        <text x="48" y="52" textAnchor="middle" fontSize="13" fontWeight="800" fill={color}>
          {value.toFixed(1)}%
        </text>
      </svg>
      <div style={{ fontSize:11, color:'#94a3b8', fontWeight:600,
        textTransform:'uppercase', letterSpacing:'.5px' }}>{label}</div>
    </div>
  );
}

/* ── Small components ── */
function KpiCard({ icon, label, value, sub, accent, hl }) {
  return (
    <div className={`kpi-card ${hl?'hl':''}`} style={{'--ac':accent}}>
      <div className="kpi-icon">{icon}</div>
      <div className="kpi-val" style={{ color:accent }}>{value}</div>
      <div className="kpi-lbl">{label}</div>
      <div className="kpi-sub">{sub}</div>
    </div>
  );
}
function Section({ title, children }) {
  return (
    <div className="form-section">
      <div className="section-title">{title}</div>
      {children}
    </div>
  );
}
function InfoRow({ label, value, hl }) {
  return (
    <div className={`info-row ${hl?'hl':''}`}>
      <span className="info-lbl">{label}</span>
      <span className="info-val" style={hl?{color:'#1d4ed8',fontSize:15}:{}}>{value}</span>
    </div>
  );
}
function Sel({ label, v, onChange, opts }) {
  return (
    <div className="field">
      <label className="field-lbl">{label}</label>
      <select className="field-ctrl" value={v} onChange={e=>onChange(e.target.value)}>
        {opts.map(o=><option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}
function Num({ label, v, onChange }) {
  return (
    <div className="field">
      <label className="field-lbl">{label}</label>
      <input className="field-ctrl" type="number" value={v} onChange={e=>onChange(e.target.value)}/>
    </div>
  );
}
function Txt({ label, v, onChange }) {
  return (
    <div className="field">
      <label className="field-lbl">{label}</label>
      <input className="field-ctrl" value={v} onChange={e=>onChange(e.target.value)}/>
    </div>
  );
}
function Spinner() { return <span className="spinner"/>; }
function GridIcon()    { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>; }
function PredictIcon() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>; }
function UsersIcon()   { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>; }
function InfoIcon()    { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>; }
function WarnIcon()    { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>; }
function ArrowIcon()   { return <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>; }
