import pytest
from fastapi.testclient import TestClient
import os
import json
import sys
import logging 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app as fastapi_app
from app import main as app_main_module
from app import generator

TEST_CONFIG = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module") 
def test_app_with_config():
    global TEST_CONFIG

    original_main_config_path = app_main_module.CONFIG_PATH
    original_main_config_state = app_main_module.CONFIG
    original_generator_config_state = generator.CONFIG
    original_generator_model_state = generator.MODEL
    original_generator_vectorizer_state = generator.VECTORIZER

    if isinstance(original_main_config_state, dict):
        original_main_config_state = dict(original_main_config_state)
    if isinstance(original_generator_config_state, dict):
        original_generator_config_state = dict(original_generator_config_state)

    try:
        logger.info("[Fixture] Setting up test environment...")
        test_config_file_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        if not os.path.exists(test_config_file_path):
            pytest.fail(f"[Fixture] Test config file not found: {test_config_file_path}")
        app_main_module.CONFIG_PATH = test_config_file_path
        logger.info(f"[Fixture] Forcing load_config from: {app_main_module.CONFIG_PATH}")
        ####
        app_main_module.CONFIG_PATH = test_config_file_path
        with TestClient(fastapi_app) as client:
            logger.info("[Fixture] TestClient created. Lifespan startup should have run.")
        #######
        TEST_CONFIG = app_main_module.CONFIG 
        ####
        if TEST_CONFIG is None:
                pytest.fail("[Fixture] TEST_CONFIG (from app_main_module.CONFIG) is None after lifespan startup.")
        # yield client
        ######
        if TEST_CONFIG is None:
            pytest.fail("[Fixture] TEST_CONFIG is None after app_main_module.load_config()")
        logger.info(f"[Fixture] TEST_CONFIG loaded: {TEST_CONFIG.get('model_name')}")
        with TestClient(fastapi_app) as client:
            logger.info("[Fixture] TestClient created. Startup event should have run.")
            if generator.MODEL is None or generator.VECTORIZER is None:
                logger.error("[Fixture] Post-startup: Model or Vectorizer is STILL None in generator.")
            else:
                logger.info("[Fixture] Post-startup: Model and Vectorizer seem loaded in generator.")
            yield client 

    except Exception as e:
        pytest.fail(f"[Fixture] Error during test setup: {e}", pytrace=True)
    finally:
        logger.info("[Fixture] Tearing down test environment...")
        app_main_module.CONFIG_PATH = original_main_config_path
        app_main_module.CONFIG = original_main_config_state
        generator.CONFIG = original_generator_config_state
        generator.MODEL = original_generator_model_state
        generator.VECTORIZER = original_generator_vectorizer_state
        TEST_CONFIG = None
        logger.info("[Fixture] Teardown complete.")

def test_health_check(test_app_with_config):
    client = test_app_with_config
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_version_endpoint(test_app_with_config):
    client = test_app_with_config
    global TEST_CONFIG
    assert TEST_CONFIG is not None, "TEST_CONFIG was not loaded for version_endpoint"
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "service_model_name" in data, f"Response data: {data}"
    assert data["service_model_name"] == TEST_CONFIG["model_name"]
    assert data["version"] == TEST_CONFIG["version"]

def test_predict_spam(test_app_with_config):
    client = test_app_with_config
    global TEST_CONFIG
    assert TEST_CONFIG is not None, "TEST_CONFIG was not loaded for predict_spam"
    assert generator.MODEL is not None, "Generator.MODEL is None at the start of test_predict_spam"
    assert generator.VECTORIZER is not None, "Generator.VECTORIZER is None at the start of test_predict_spam"

    payload = {"post": "URGENT! You have won 10,000 INR! Click here!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Predict failed with {response.status_code}, content: {response.text}"
    data = response.json()
    assert "label" in data
    assert data["label"] in TEST_CONFIG["labels"]

def test_predict_not_spam(test_app_with_config):
    client = test_app_with_config
    global TEST_CONFIG
    assert TEST_CONFIG is not None
    assert generator.MODEL is not None
    assert generator.VECTORIZER is not None
    payload = {"post": "Hello, how are you doing today?"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Predict failed with {response.status_code}, content: {response.text}"
    data = response.json()
    assert "label" in data
    assert data["label"] in TEST_CONFIG["labels"]

def test_predict_empty_input_string(test_app_with_config):
    client = test_app_with_config
    global TEST_CONFIG
    assert TEST_CONFIG is not None
    assert generator.MODEL is not None
    assert generator.VECTORIZER is not None
    payload = {"post": " "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Predict failed with {response.status_code}, content: {response.text}"
    data = response.json()
    assert data["label"] == TEST_CONFIG["labels"][0]

def test_predict_output_format(test_app_with_config):
    client = test_app_with_config
    global TEST_CONFIG
    assert TEST_CONFIG is not None
    assert generator.MODEL is not None
    assert generator.VECTORIZER is not None
    payload = {"post": "Test schema output"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Predict failed with {response.status_code}, content: {response.text}"
    data = response.json()
    assert isinstance(data["label"], str)
    assert data["label"] in TEST_CONFIG["labels"]

def test_predict_invalid_input_missing_field(test_app_with_config):
    client = test_app_with_config
    payload = {"text": "This is missing the 'post' field"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_input_wrong_type(test_app_with_config):
    client = test_app_with_config
    payload = {"post": 12345}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422