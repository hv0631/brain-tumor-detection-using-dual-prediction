# Brain Tumor Detection API

A FastAPI-based REST API wrapping the Dual Prediction Brain Tumor Detection system.

---

## Project Structure

```
brain_tumor_api/
├── main.py                  ← FastAPI app (this file)
├── detector.py              ← Your original DualPredictionBrainTumorDetector class
├── requirements.txt
├── models/                  ← Saved model files go here (auto-created)
│   ├── brain_tumor_csv_model.pkl
│   ├── brain_tumor_image_model.h5
│   └── brain_tumor_scaler.pkl
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your detector code
Copy your original Python file and rename it to `detector.py` in the same folder.

### 3. Run the API locally
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be live at: http://localhost:8000  
Swagger docs at:     http://localhost:8000/docs  
ReDoc at:            http://localhost:8000/redoc  

---

## API Endpoints

| Method | Endpoint         | Description                              |
|--------|------------------|------------------------------------------|
| GET    | `/`              | API info and available endpoints         |
| GET    | `/health`        | Check if models are loaded               |
| GET    | `/model-info`    | Model architecture and parameters        |
| POST   | `/predict`       | Upload image → get tumor prediction      |
| POST   | `/train`         | Trigger training on image folder         |
| POST   | `/reload-models` | Reload saved models from disk            |

---

## Usage Examples

### Check health
```bash
curl http://localhost:8000/health
```

### Predict on an image
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/brain_scan.jpg"
```

### Train models
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "image_folder": "C:/Users/hvver/Desktop/ntcc/data/Brian/Training",
    "model_type": "random_forest",
    "epochs": 30,
    "batch_size": 32
  }'
```

### Sample Predict Response
```json
{
  "filename": "scan001.jpg",
  "csv_model": {
    "prediction": 1,
    "label": "TUMOR",
    "tumor_probability": 0.87,
    "confidence": 0.87
  },
  "cnn_model": {
    "prediction": 1,
    "label": "TUMOR",
    "tumor_probability": 0.91,
    "confidence": 0.91
  },
  "ensemble": {
    "prediction": 1,
    "label": "TUMOR",
    "tumor_probability": 0.89,
    "confidence": 0.89
  },
  "features": {
    "area": 12453.0,
    "perimeter": 512.3,
    "compactness": 0.473,
    "contrast": 48.21,
    "energy": 0.312,
    "homogeneity": 0.021,
    "entropy": 6.84,
    "mean_intensity": 104.3,
    "std_intensity": 48.21
  }
}
```

---

## Deployment

### Option A — Render.com (Free, recommended for demos)
1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo
4. Set Build Command: `pip install -r requirements.txt`
5. Set Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Done!

### Option B — Railway.app (Free tier)
1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Option C — Hugging Face Spaces
1. Create a new Space with SDK = "Docker"
2. Add a `Dockerfile` pointing to uvicorn

---

## Notes
- If models are not found on startup, the API still runs but `/predict` returns 503.
- Use `POST /train` to train, then `GET /health` to poll until models are ready.
- If you already have `.pkl` and `.h5` files, place them in the `models/` folder and call `POST /reload-models`.
