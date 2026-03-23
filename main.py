import warnings
warnings.filterwarnings("ignore")

import os
import io
import numpy as np
import pandas as pd
import cv2
import joblib
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import tensorflow as tf
from tensorflow import keras

# ── app setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Detection API",
    description="Dual-model brain tumor detection using CNN + Random Forest ensemble.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CSV_MODEL_PATH  = BASE_DIR / "models" / "brain_tumor_csv_model.pkl"
IMAGE_MODEL_PATH= BASE_DIR / "models" / "brain_tumor_image_model.h5"
SCALER_PATH     = BASE_DIR / "models" / "brain_tumor_scaler.pkl"

# ── global model state ────────────────────────────────────────────────────────
class ModelState:
    csv_model   = None
    image_model = None
    scaler      = None
    loaded      = False
    load_error  = None

state = ModelState()

FEATURE_COLUMNS = [
    "area", "perimeter", "compactness", "contrast",
    "energy", "homogeneity", "entropy", "mean_intensity", "std_intensity",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_models():
    """Attempt to load saved models from disk."""
    missing = []
    for p in [CSV_MODEL_PATH, IMAGE_MODEL_PATH, SCALER_PATH]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        state.load_error = f"Model files not found: {missing}. Please train first via POST /train."
        state.loaded = False
        return False

    try:
        state.csv_model   = joblib.load(CSV_MODEL_PATH)
        state.image_model = keras.models.load_model(IMAGE_MODEL_PATH)
        state.scaler      = joblib.load(SCALER_PATH)
        state.loaded      = True
        state.load_error  = None
        return True
    except Exception as e:
        state.load_error = str(e)
        state.loaded     = False
        return False


def extract_features(image_array: np.ndarray) -> Optional[dict]:
    """Extract handcrafted features from a grayscale image array (256×256)."""
    try:
        image = cv2.resize(image_array, (256, 256))
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            lc = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(lc)
            perimeter = cv2.arcLength(lc, True)
            compactness = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0
        else:
            area = perimeter = compactness = 0

        mean_i = float(np.mean(image))
        std_i  = float(np.std(image))
        contrast = std_i

        norm = image.astype(np.float32) / 255.0
        energy = float(np.sum(norm ** 2) / (256 * 256))

        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        homogeneity = float(1.0 / (1.0 + np.mean(np.sqrt(gx**2 + gy**2))))

        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist[hist > 0].astype(float)
        hist /= hist.sum()
        entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0

        return {
            "area": float(area), "perimeter": float(perimeter),
            "compactness": float(compactness), "contrast": contrast,
            "energy": energy, "homogeneity": homogeneity,
            "entropy": entropy, "mean_intensity": mean_i, "std_intensity": std_i,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")


def preprocess_for_cnn(image_array: np.ndarray) -> np.ndarray:
    """Resize, normalise, and add channel dim for CNN."""
    img = cv2.resize(image_array, (128, 128)).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=-1)


def bytes_to_gray(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes to grayscale numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Ensure it is a valid JPG/PNG.")
    return img


# ── startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    (BASE_DIR / "models").mkdir(exist_ok=True)
    load_models()


# ── response schemas ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    message: str

class PredictionResponse(BaseModel):
    filename: str
    csv_model: dict
    cnn_model: dict
    ensemble: dict
    features: dict

class TrainRequest(BaseModel):
    image_folder: str
    model_type: str = "random_forest"   # random_forest | svm | logistic_regression
    epochs: int = 20
    batch_size: int = 32
    test_size: float = 0.2

class TrainResponse(BaseModel):
    status: str
    message: str
    csv_accuracy: Optional[float] = None
    image_accuracy: Optional[float] = None
    ensemble_accuracy: Optional[float] = None


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "api": "Brain Tumor Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/train", "/model-info", "/reload-models"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    if state.loaded:
        return HealthResponse(status="ok", models_loaded=True, message="Models loaded and ready.")
    return HealthResponse(
        status="warning",
        models_loaded=False,
        message=state.load_error or "Models not loaded. Use POST /train to train them.",
    )


@app.get("/model-info", tags=["Models"])
def model_info():
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Train first via POST /train.")

    csv_info = {
        "type": type(state.csv_model).__name__,
        "params": str(state.csv_model.get_params()) if hasattr(state.csv_model, "get_params") else "N/A",
    }

    cnn_layers = []
    if state.image_model:
        for layer in state.image_model.layers:
            cnn_layers.append({"name": layer.name, "type": layer.__class__.__name__})

    return {
        "csv_model": csv_info,
        "cnn_model": {
            "total_layers": len(state.image_model.layers) if state.image_model else 0,
            "layers": cnn_layers,
            "total_params": state.image_model.count_params() if state.image_model else 0,
        },
        "feature_columns": FEATURE_COLUMNS,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a brain scan image (JPG / PNG) and get predictions from:
    - CSV / feature-based Random Forest model
    - CNN image model
    - Ensemble (average of both probabilities)
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Train first via POST /train.")

    # Read & decode
    data = await file.read()
    gray = bytes_to_gray(data)

    # ── CSV model ──
    features = extract_features(gray)
    feat_df  = pd.DataFrame([features])[FEATURE_COLUMNS]

    try:
        feat_scaled = state.scaler.transform(feat_df)
    except Exception:
        feat_scaled = feat_df

    csv_prob_arr = state.csv_model.predict_proba(feat_scaled)[0]
    csv_pred     = int(np.argmax(csv_prob_arr))
    csv_prob     = float(csv_prob_arr[1])
    csv_conf     = float(csv_prob_arr[csv_pred])

    # ── CNN model ──
    cnn_input    = preprocess_for_cnn(gray)
    cnn_batch    = np.expand_dims(cnn_input, axis=0)
    cnn_prob     = float(state.image_model.predict(cnn_batch, verbose=0)[0][0])
    cnn_pred     = int(cnn_prob > 0.5)
    cnn_conf     = cnn_prob if cnn_pred == 1 else (1 - cnn_prob)

    # ── Ensemble ──
    ens_prob = (csv_prob + cnn_prob) / 2
    ens_pred = int(ens_prob > 0.5)
    ens_conf = ens_prob if ens_pred == 1 else (1 - ens_prob)

    label = lambda p: "TUMOR" if p == 1 else "NO TUMOR"

    return PredictionResponse(
        filename=file.filename,
        csv_model={
            "prediction": csv_pred,
            "label": label(csv_pred),
            "tumor_probability": round(csv_prob, 4),
            "confidence": round(csv_conf, 4),
        },
        cnn_model={
            "prediction": cnn_pred,
            "label": label(cnn_pred),
            "tumor_probability": round(cnn_prob, 4),
            "confidence": round(cnn_conf, 4),
        },
        ensemble={
            "prediction": ens_pred,
            "label": label(ens_pred),
            "tumor_probability": round(ens_prob, 4),
            "confidence": round(ens_conf, 4),
        },
        features={k: round(v, 4) for k, v in features.items()},
    )


@app.post("/train", response_model=TrainResponse, tags=["Training"])
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger model training in the background.
    Provide the absolute path to your image folder on the server.
    """
    if not os.path.isdir(req.image_folder):
        raise HTTPException(status_code=400, detail=f"Folder not found: {req.image_folder}")

    def run_training():
        try:
            # Import here to avoid circular issues
            import sys
            sys.path.insert(0, str(BASE_DIR.parent))
            from detector import DualPredictionBrainTumorDetector

            (BASE_DIR / "models").mkdir(exist_ok=True)

            detector = DualPredictionBrainTumorDetector(req.image_folder)
            detector.process_images_to_csv(str(BASE_DIR / "brain_tumor_features.csv"))
            detector.prepare_data_splits(test_size=req.test_size)
            detector.train_csv_model(model_type=req.model_type)
            detector.train_image_model(epochs=req.epochs, batch_size=req.batch_size)
            detector.save_trained_models(
                csv_model_path=str(CSV_MODEL_PATH),
                image_model_path=str(IMAGE_MODEL_PATH),
                scaler_path=str(SCALER_PATH),
            )
            load_models()
            print("✅ Training complete. Models saved and reloaded.")
        except Exception as e:
            print(f"❌ Training failed: {e}")

    background_tasks.add_task(run_training)
    return TrainResponse(
        status="started",
        message="Training started in background. Poll GET /health to check when models are ready.",
    )


@app.post("/reload-models", tags=["Models"])
def reload_models():
    """Force reload models from disk (useful after external training)."""
    success = load_models()
    if success:
        return {"status": "ok", "message": "Models reloaded successfully."}
    raise HTTPException(status_code=500, detail=state.load_error)
