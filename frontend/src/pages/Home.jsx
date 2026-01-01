import React from 'react';
import { useNavigate } from 'react-router-dom';

const capture = [
  {
    title: 'Input sources',
    items: ['Face + voice + video', 'Vitals: HR, BP, temp, SpO2 (optional)', 'Symptoms + questionnaire text'],
  },
  {
    title: 'Processing layer',
    items: ['OpenCV/Mediapipe alignment & quality gating', 'Frame sampling and denoise', 'Secure upload channel'],
  },
];

const parallelIntelligence = [
  { title: 'CNN phenotyping', detail: 'Micro-patterns for distress, fatigue, pain cues.' },
  { title: 'LSTM temporal analysis', detail: 'Frame-to-frame shifts in expressions and breathing effort.' },
  { title: 'Transformer NLP', detail: 'Symptom parsing and lightweight clinical keyword expansion.' },
  { title: 'Video frame masks', detail: 'Region focus for temporal saliency (demo-grade).' },
];

const explainability = [
  { label: 'Grad-CAM heatmap', text: 'Visual layer to show where the face model focused.' },
  { label: 'SHAP / LIME', text: 'On-demand feature-level contributions for transparency.' },
  { label: 'Counterfactuals', text: 'Recovery-style suggestions to reason about change.' },
];

const outputs = [
  'Risk level and numeric score with confidence band.',
  'Modal contributions to see influence balance.',
  'Suggested next steps phrased for discussion, not orders.',
  'FHIR Observation JSON export for integration exercises.',
];

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="home">
      <div className="hero-modern">
        <div className="hero-content">
          <div className="eyebrow-modern">
            <span className="eyebrow-icon">🧬</span>
            Multimodal · Explainable · AI-Powered
          </div>
          <h1 className="hero-title">HealthMorph AI</h1>
          <p className="hero-subtitle">Advanced health risk assessment through facial analysis, voice patterns, and symptom intelligence</p>
          <div className="hero-actions">
            <button className="button-primary" onClick={() => navigate('/analyze')}>
              <span className="button-icon">🔬</span>
              Start Analysis
            </button>
          </div>
        </div>
        <div className="hero-visual">
          <div className="floating-card card-1">🧠 Neural Analysis</div>
          <div className="floating-card card-2">📊 Risk Assessment</div>
          <div className="floating-card card-3">🎯 Pattern Detection</div>
        </div>
      </div>

      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-icon">📸</div>
          <h3>Data Capture & Processing</h3>
          <div className="feature-sections">
            {capture.map((block) => (
              <div key={block.title} className="feature-section">
                <div className="section-title">{block.title}</div>
                <ul className="feature-list-modern">
                  {block.items.map((item) => (
                    <li key={item}>
                      <span className="check-icon">✓</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <div className="feature-card">
          <div className="feature-icon">🤖</div>
          <h3>AI Intelligence Layer</h3>
          <div className="intelligence-grid">
            {parallelIntelligence.map((item) => (
              <div key={item.title} className="intelligence-card">
                <div className="intelligence-title">{item.title}</div>
                <div className="intelligence-detail">{item.detail}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="features-grid">
        <div className="feature-card explainability">
          <div className="feature-icon">🔍</div>
          <h3>Explainability & Transparency</h3>
          <div className="explainability-grid">
            {explainability.map((item) => (
              <div key={item.label} className="explainability-card">
                <div className="explainability-label">{item.label}</div>
                <div className="explainability-text">{item.text}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="feature-card outputs">
          <div className="feature-icon">📋</div>
          <h3>Analysis Results</h3>
          <ul className="outputs-list">
            {outputs.map((item) => (
              <li key={item}>
                <span className="output-icon">→</span>
                {item}
              </li>
            ))}
          </ul>
          <div className="fusion-card">
            <div className="fusion-title">🔬 Weighted Fusion Analysis</div>
            <p className="fusion-text">Advanced multimodal integration combines facial patterns, symptom analysis, and vital signs to generate comprehensive risk assessment with visual heatmaps and detailed feature attribution.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
