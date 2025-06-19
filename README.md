# Spam Detector Microservice

A production-grade, containerized microservice that classifies text-based inputs as `spam` or `not_spam`, along with a confidence score.

## Features

*   Runs fully offline.
*   Accepts input strictly via JSON.
*   Standard REST API interface.
*   Deployed as a Docker container.
*   Driven by a runtime `config.json`.
*   Trained on India-region-specific data (placeholder used in this example, replace with real data).
*   Uses TF-IDF vectorization and Naive Bayes (or Logistic Regression) classifier.

## Project Structure
```bash
spam-detector/
├── app/ # Main application code
│ ├── main.py # FastAPI app, API routing
│ ├── generator.py # Model prediction logic, loading model
│ └── model/ # Trained model + vectorizer (e.g., .pkl files)
├── data/ # Datasets
│ └── dataset.csv # Labeled input dataset (needs to be India-specific)
├── tests/ # Unit tests
│ └── test_predict.py
├── config.json # Runtime settings (mandatory)
├── Dockerfile # For Docker deployment
├── requirements.txt # Python dependencies
├── README.md # This file
├── schema.json # JSON input/output structure documentation
└── train_model.py # Script to train the model
```
## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd spam-detector
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Crucial) Prepare your Dataset:**
    Replace `data/dataset.csv` with your own India-region-specific dataset. It must have 'text' and 'label' columns, with labels being 'spam' or 'not_spam'. A minimum of 2000 balanced entries is recommended.

5.  **Train the Model:**
    Run the training script. This will preprocess the data, train the TF-IDF vectorizer and classifier, and save them to `app/model/`.
    ```bash
    python train_model.py
    ```
    *Note: You can choose between 'naive_bayes' and 'logistic_regression' by editing `MODEL_CHOICE` in `train_model.py`. Ensure `config.json` reflects the chosen model file name.*

## Running the Service Locally

Once the model is trained and dependencies are installed:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000.

## API Endpoints

* POST /predict: Classifies a given text.
> Payload: {"post": "Your text here..."}

>Response: {"label": "spam/not_spam", "confidence": 0.xx}
* GET /health: Health check.
>Response: {"status": "ok"}
* GET /version: Returns model version and config metadata.
>Response: {"model_name": "...", "version": "...", "config_details": {...}}

## Dockerization

* Build the Docker image:
Ensure Dockerfile, config.json, requirements.txt, and the app/ and data/ (with trained model in app/model/) directories are present.

```bash
docker build -t spam-detector .
```

Run the Docker container:
```bash
docker run -p 8000:8000 spam-detector
```

The service will be accessible at http://localhost:8000 on your host machine. The service runs entirely offline inside the container.