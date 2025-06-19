import json
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator 
import logging
from contextlib import asynccontextmanager 

from . import generator
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.json')
CONFIG = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config_and_models():
    """
    Loads configuration and then loads the model and vectorizer.
    This function will be called during the 'startup' phase of the lifespan.
    """
    global CONFIG 
    logger.info("Lifespan startup: Attempting to load config...")
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"CRITICAL: Configuration file '{CONFIG_PATH}' not found. Application cannot start.")
        raise RuntimeError(f"Configuration file '{CONFIG_PATH}' not found.")
    try:
        with open(CONFIG_PATH, 'r') as f:
            CONFIG = json.load(f) 
        logger.info("Configuration loaded successfully.")
        required_keys = ["model_name", "version", "labels", "model_file", "vectorizer_file", "model_path_prefix"]
        for key in required_keys:
            if key not in CONFIG:
                logger.error(f"CRITICAL: Missing key '{key}' in config.json. Application cannot start.")
                raise ValueError(f"Missing key '{key}' in config.json")
        if not isinstance(CONFIG.get("labels"), list) or len(CONFIG.get("labels")) != 2:
            logger.error("CRITICAL: 'labels' in config.json must be a list of two strings. Application cannot start.")
            raise ValueError("'labels' in config.json must be a list of two strings.")

     
        logger.info(f"Lifespan startup: Config loaded. Model path prefix: {CONFIG.get('model_path_prefix')}, Model file: {CONFIG.get('model_file')}")
        generator.load_model_and_vectorizer(CONFIG) 
        logger.info("Lifespan startup: Model and vectorizer loading process initiated by generator.")

        if generator.MODEL is None or generator.VECTORIZER is None:
            logger.error("Lifespan startup: CRITICAL - Model or Vectorizer is still None after generator.load_model_and_vectorizer call.")
           
        else:
            logger.info("Lifespan startup: Model and vectorizer appear to be loaded successfully by generator.")

    except json.JSONDecodeError as e:
        logger.error(f"CRITICAL: Invalid JSON in '{CONFIG_PATH}': {e}. Application cannot start.")
        raise RuntimeError(f"Invalid JSON in '{CONFIG_PATH}': {e}")
    except ValueError as e: 
        logger.error(f"CRITICAL: Configuration error: {e}. Application cannot start.")
        raise RuntimeError(f"Configuration error: {e}")
    except FileNotFoundError as e: 
        logger.error(f"CRITICAL: Model/Vectorizer file not found: {e}. Application cannot start.")
        raise RuntimeError(f"Model/Vectorizer file not found: {e}")
    except Exception as e:
        logger.error(f"CRITICAL ERROR during startup (loading config/models): {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed due to critical error: {e}")



@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Application startup sequence initiated via lifespan...")
    try:
        load_config_and_models()
        logger.info("Application startup sequence completed successfully.")
    except Exception as e:

        logger.critical(f"Lifespan startup FAILED: {e}. Application will likely not serve requests correctly.", exc_info=True)

        raise
    
    yield 
    logger.info("Application shutdown sequence initiated via lifespan...")
    global CONFIG, generator
    CONFIG = None
    generator.MODEL = None
    generator.VECTORIZER = None
    logger.info("Global states reset during shutdown.")
    logger.info("Application shutdown sequence completed.")


app = FastAPI(title="Spam Detector API", version="1.0.0", lifespan=lifespan)


class TextInput(BaseModel):
    post: str = Field(..., min_length=1, description="The text content to classify (e.g., post or comment).")

class PredictionOutput(BaseModel):
    label: str = Field(..., description="Classification label, e.g., 'spam' or 'not_spam'.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the prediction.")

    @field_validator('label')
    @classmethod
    def label_must_be_in_config(cls, value):

        
        if CONFIG and value not in CONFIG.get('labels', []):
            configured_labels = CONFIG.get('labels', []) if CONFIG else []
            raise ValueError(f"Label '{value}' is not one of the configured labels: {configured_labels}")
        return value

class HealthResponse(BaseModel):
    status: str = "ok"

class VersionResponse(BaseModel):
    service_model_name: str
    version: str
    config_details: dict

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: TextInput):
    if generator.MODEL is None or generator.VECTORIZER is None:
        logger.error("PREDICT ENDPOINT: Model or vectorizer not loaded. This indicates a startup failure.")
        raise HTTPException(status_code=503, detail="Model not ready. Please try again later or check server logs.")
    try:
        prediction_result = generator.predict_spam(data.post)
        return PredictionOutput(**prediction_result)
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()

@app.get("/version", response_model=VersionResponse)
async def get_version():
    if not CONFIG: 
        logger.error("VERSION ENDPOINT: Configuration not loaded. This indicates a startup failure.")
        raise HTTPException(status_code=503, detail="Configuration not loaded. Service may be starting or encountered an error.")
    return VersionResponse(
        service_model_name=CONFIG.get("model_name", "N/A"),
        version=CONFIG.get("version", "N/A"),
        config_details={
            "labels": CONFIG.get("labels"),
            "confidence_threshold": CONFIG.get("confidence_threshold")
        }
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Outgoing response: {response.status_code}")
    return response