# HealthMorph AI

"Insights from Within"

HealthMorph AI is an academic MVP clinical decision support web application. It analyzes a face image and self-reported symptoms to produce an early health risk indication with a simple explainable heatmap. This system does not provide medical diagnosis.

## Project Structure

```
HealthMorph-AI/
├── frontend/
│   ├── index.html
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── App.js
│   │   ├── index.js
│   └── package.json
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── preprocess.py
│   ├── explain.py
│   └── static/
├── models/
│   └── README.txt
├── requirements.txt
└── README.md
```

## Backend (FastAPI)

### Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- Static heatmaps are served at `http://localhost:8000/static/...`.
- CORS allows `http://localhost:5173` (Vite dev server).

## Frontend (React + Vite)

### Install dependencies

```bash
cd frontend
npm install
```

### Run frontend

```bash
npm run dev
```

Open the shown URL (typically `http://localhost:5173`).

## Usage

1. Open the frontend, go to Analyze.
2. Upload a face image (clear frontal face works best).
3. Enter symptoms (e.g., "moderate fever and persistent cough"). Optionally add Questionnaire text.
4. Optionally upload voice (WAV) and video (MP4) files (camera capture supported) and enter vitals (heart rate, BP, temperature, SpO2).
5. Submit and view the risk indication, heatmap (face-focused overlay), confidence, modal contributions, and suggested next steps. You can export a FHIR Observation JSON from the Results page.

## Notes & Features

- AI logic is simulated for MVP; facial preprocessing uses OpenCV face detection + normalization, and the heatmap is a Grad-CAM-style Gaussian overlay. Multimodal fusion combines facial score, symptom text, optional voice/video, and sensor vitals.
- Medical Disclaimer: This system is for academic and research purposes only and does not provide medical diagnosis.
- Built for local demo; no persistence or authentication.

### API Endpoints

- `POST /analyze`: multipart form-data
	- image (required), symptoms (required)
	- questionnaire (optional text), voice (optional WAV), video (optional MP4), sensors (optional JSON)
	- returns risk_level, risk_score, confidence, next_steps, contributions, heatmap_url
- `POST /export/fhir`: returns a minimal FHIR Observation JSON for risk score export

### Persistence, Auth, Limits

- Optional JWT auth: set `HM_AUTH_REQUIRED=true` and `HM_JWT_SECRET` for bearer token validation
- Persistence: SQLAlchemy with `HM_DB_URL` (defaults to SQLite). Disable with `HM_PERSIST_ENABLED=false`.
- Rate limiting: configurable via `HM_RATE_LIMIT_PER_MIN` (default 60/min, in-process).

### Media Retention

- Media retention is off by default. Toggle with `HM_RETAIN_MEDIA=true` (heatmaps are saved for display; originals are not retained by default).

### Privacy

- Inputs are processed in-memory and not stored; the heatmap is a static placeholder.

