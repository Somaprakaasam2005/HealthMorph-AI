import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Disclaimer from '../components/Disclaimer.jsx';

const API_BASE = 'http://localhost:8000';

export default function Results() {
  const { state } = useLocation();
  const navigate = useNavigate();
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainError, setExplainError] = useState('');
  const [shap, setShap] = useState(null);
  const [lime, setLime] = useState(null);
  const [counterfactual, setCounterfactual] = useState(null);
  const [timestamp] = useState(new Date().toISOString());

  if (!state) {
    return (
      <div className="card">
        <p>No results found. Please run an analysis.</p>
        <button className="button" onClick={() => navigate('/analyze')}>Go to Analyze</button>
      </div>
    );
  }

  const { risk_level, risk_score, explanation, heatmap_url, disclaimer, confidence, next_steps, sensors, contributions } = state;

  const buildFeaturesPayload = () => {
    if (contributions && Object.keys(contributions).length > 0) {
      const vals = Object.values(contributions).map(Number).filter((v) => !Number.isNaN(v));
      if (vals.length > 0) return JSON.stringify(vals);
    }
    return JSON.stringify([Number(risk_score) || 0]);
  };

  const requestExplain = async () => {
    setExplainError('');
    setExplainLoading(true);
    const features_json = buildFeaturesPayload();
    try {
      const formShap = new FormData();
      formShap.append('features_json', features_json);
      const shapRes = await fetch(`${API_BASE}/explain/shap`, { method: 'POST', body: formShap });
      if (!shapRes.ok) throw new Error('SHAP request failed');
      setShap(await shapRes.json());

      const formLime = new FormData();
      formLime.append('features_json', features_json);
      const limeRes = await fetch(`${API_BASE}/explain/lime`, { method: 'POST', body: formLime });
      if (!limeRes.ok) throw new Error('LIME request failed');
      setLime(await limeRes.json());

      const formCf = new FormData();
      formCf.append('features_json', features_json);
      formCf.append('analysis_type', 'recovery');
      const cfRes = await fetch(`${API_BASE}/explain/counterfactual`, { method: 'POST', body: formCf });
      if (!cfRes.ok) throw new Error('Counterfactual request failed');
      setCounterfactual(await cfRes.json());
    } catch (err) {
      setExplainError(err.message || 'Explainability request failed');
    } finally {
      setExplainLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Results</h2>
      <div className="grid">
        <div>
          <h3>Risk Level: {risk_level}</h3>
          <p><strong>Risk Score:</strong> {risk_score}%</p>
          {typeof confidence === 'number' && <p><strong>Confidence:</strong> {confidence}%</p>}
          <p>{explanation}</p>
          {next_steps && <p><strong>Suggested Next Steps:</strong> {next_steps}</p>}
          {contributions && (
            <p><strong>Modal Contributions:</strong> {Object.entries(contributions).map(([k, v]) => `${k}: ${v}`).join(', ')}</p>
          )}
          {sensors && Object.keys(sensors).length > 0 && (
            <p><strong>Sensor Summary:</strong> {Object.entries(sensors).filter(([_, v]) => v !== undefined).map(([k, v]) => `${k}: ${v}`).join(', ')}</p>
          )}
          <button className="button secondary" onClick={() => navigate('/analyze')}>Re-analyze</button>
          <button className="button" style={{ marginLeft: 8 }} onClick={requestExplain} disabled={explainLoading}>
            {explainLoading ? 'Generating explainability…' : 'Generate explainability'}
          </button>
          {explainError && <p style={{ color: '#b91c1c', marginTop: 8 }}>{explainError}</p>}
        </div>
        <div>
          {heatmap_url ? (
            <img src={`${API_BASE}${heatmap_url}`} alt="Explainable AI heatmap" style={{ width: '100%', borderRadius: 8, border: '1px solid #e5e7eb' }} />
          ) : (
            <p>Heatmap not available.</p>
          )}
        </div>
      </div>
      <div className="disclaimer"><strong>Medical Disclaimer:</strong> {disclaimer || 'This system is for academic and research purposes only and does not provide medical diagnosis.'}</div>
      {(shap || lime || counterfactual) && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Explainability</h3>
          {shap && shap.shap && (
            <div style={{ marginBottom: 12 }}>
              <strong>SHAP Top Features:</strong>{' '}
              {(shap.shap.importances || []).map((imp) => imp.feature_name).join(', ') || 'n/a'}
            </div>
          )}
          {lime && lime.lime && (
            <div style={{ marginBottom: 12 }}>
              <strong>LIME Local Factors:</strong>{' '}
              {(lime.lime.local_features || []).map(([feat, weight]) => `${feat} (${weight.toFixed(2)})`).join(', ') || 'n/a'}
            </div>
          )}
          {counterfactual && counterfactual.counterfactual && (
            <div>
              <strong>Counterfactual Suggestion:</strong>{' '}
              {counterfactual.counterfactual.recommendations
                ? counterfactual.counterfactual.recommendations.map((r) => r.feature).join(', ')
                : 'See recovery/boundary details'}
            </div>
          )}
        </div>
      )}
      <div className="export-group" style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button className="button" onClick={async () => {
          try {
            const form = new FormData();
            form.append('risk_level', risk_level);
            form.append('risk_score', risk_score);
            form.append('subject_id', 'subject-001');
            const res = await fetch(`${API_BASE}/export/fhir`, { method: 'POST', body: form });
            if (!res.ok) throw new Error('Export failed');
            const data = await res.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'healthmorph-observation.json';
            a.click();
            URL.revokeObjectURL(url);
          } catch (err) {
            alert(err.message || 'FHIR export failed');
          }
        }}>Export FHIR</button>
        <button className="button secondary" onClick={() => {
          const csv = [
            ['Field', 'Value'],
            ['Risk Level', risk_level],
            ['Risk Score', risk_score],
            ['Confidence', confidence],
            ['Timestamp', timestamp],
            ['Explanation', explanation],
          ].map(row => row.map(cell => `"${cell || ''}"`).join(',')).join('\n');
          const blob = new Blob([csv], { type: 'text/csv' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'healthmorph-results.csv';
          a.click();
          URL.revokeObjectURL(url);
        }}>Export CSV</button>
        <button className="button secondary" onClick={() => {
          const data = {
            risk_level, risk_score, confidence, explanation, next_steps,
            contributions, sensors, timestamp, heatmap_url
          };
          const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'healthmorph-raw.json';
          a.click();
          URL.revokeObjectURL(url);
        }}>Export JSON</button>
      </div>

      <div className="card analytics-dash" style={{ marginTop: 16 }}>
        <div className="eyebrow">Analytics dashboard</div>
        <h3 style={{ margin: '6px 0 14px' }}>Detailed breakdown</h3>
        <div className="analytics-grid">
          <div className="metric-card">
            <div className="metric-label">Risk Score</div>
            <div className="metric-value">{risk_score ?? '--'}%</div>
            <div className="metric-bar">
              <div className="metric-fill" style={{ width: `${Math.min(risk_score || 0, 100)}%` }} />
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Confidence</div>
            <div className="metric-value">{confidence ?? '--'}%</div>
            <div className="metric-bar">
              <div className="metric-fill conf" style={{ width: `${Math.min(confidence || 0, 100)}%` }} />
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Risk Band</div>
            <div className="metric-badge">{risk_level || 'n/a'}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Session</div>
            <div className="metric-time">{new Date(timestamp).toLocaleTimeString()}</div>
          </div>
        </div>
      </div>

      <div className="card report-preview" style={{ marginTop: 16 }}>
        <div className="report-header">
          <div>
            <div className="eyebrow">Report preview</div>
            <h3 style={{ margin: '6px 0' }}>AI Bio-Assist Summary</h3>
            <p style={{ margin: 0, color: '#475569' }}>Fusion score and per-modality influence at a glance.</p>
            <div className="freshness">Session-only preview · Export to persist</div>
          </div>
          <div className="score-badge">
            <div className="score-value">{risk_score ?? '--'}%</div>
            <div className="score-label">Risk score</div>
          </div>
        </div>

        <div className="report-grid">
          <div className="report-tile">
            <div className="tile-label">Risk level</div>
            <div className="tile-value">{risk_level || 'n/a'}</div>
            {typeof confidence === 'number' && <div className="tile-meta">Confidence: {confidence}%</div>}
          </div>
          <div className="report-tile">
            <div className="tile-label">Next steps</div>
            <div className="tile-meta">{next_steps || 'Review results with a human expert; do not self-diagnose.'}</div>
          </div>
        </div>

        <div className="mini-chart">
          <div className="tile-label">Modal contributions</div>
          <div className="bar-grid">
            {contributions && Object.keys(contributions).length > 0 ? (
              Object.entries(contributions).map(([k, v]) => {
                const value = Number(v) || 0;
                const width = Math.min(Math.abs(value), 100);
                return (
                  <div key={k} className="bar-row">
                    <span className="bar-label">{k}</span>
                    <div className="bar-track">
                      <div className={`bar-fill ${value >= 0 ? 'pos' : 'neg'}`} style={{ width: `${width}%` }} />
                    </div>
                    <span className="bar-value">{value.toFixed ? value.toFixed(1) : value}</span>
                  </div>
                );
              })
            ) : (
              <div className="tile-meta">No contributions provided.</div>
            )}
          </div>
          <div className="legend">
            <span className="legend-dot pos" /> Positive influence
            <span className="legend-dot neg" /> Negative influence
          </div>
        </div>

        <div className="callout warn">
          <div className="callout-title">Important</div>
          <div className="callout-text">High scores alone are not diagnostic. Combine with clinical judgment and verified devices.</div>
        </div>
      </div>
    </div>
  );
}
