import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.word_tokenize("test")
except LookupError:
    nltk.download('punkt', quiet=True)
    


MODEL = None
VECTORIZER = None
CONFIG = None 
PREPROCESS_PIPELINE = None 

stop_words_global = set(stopwords.words('english'))
stemmer_global = PorterStemmer() 

def preprocess_text_inference(text):
    """Preprocessing function consistent with training."""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    # text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in text_tokens if word not in stop_words_global and word.isalpha()]
    return " ".join(filtered_text)

import logging
logger = logging.getLogger(__name__) 

def load_model_and_vectorizer(config_data):
    global MODEL, VECTORIZER, CONFIG, PREPROCESS_PIPELINE
    CONFIG = config_data
    logger.info(f"Generator: Attempting to load model with config: {CONFIG}")
    
    model_full_path = os.path.join(CONFIG['model_path_prefix'], CONFIG['model_file'])
    vectorizer_full_path = os.path.join(CONFIG['model_path_prefix'], CONFIG['vectorizer_file'])

    logger.info(f"Generator: Expected model path: {model_full_path}")
    logger.info(f"Generator: Expected vectorizer path: {vectorizer_full_path}")

    if not os.path.exists(model_full_path):
        logger.error(f"Generator: Model file not found: {model_full_path}")
        raise FileNotFoundError(f"Model file not found: {model_full_path}")
    if not os.path.exists(vectorizer_full_path):
        logger.error(f"Generator: Vectorizer file not found: {vectorizer_full_path}")
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_full_path}")
        
    try:
        MODEL = joblib.load(model_full_path)
        VECTORIZER = joblib.load(vectorizer_full_path)
        PREPROCESS_PIPELINE = preprocess_text_inference
        logger.info(f"Generator: Model '{CONFIG['model_name']}' and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Generator: Error loading model/vectorizer from disk: {e}", exc_info=True)
        MODEL = None 
        VECTORIZER = None
        raise


def predict_spam(text_input: str):
    if MODEL is None or VECTORIZER is None or PREPROCESS_PIPELINE is None:
        raise RuntimeError("Model, vectorizer, or preprocessing pipeline not loaded. Call load_model_and_vectorizer first.")

    processed_text = PREPROCESS_PIPELINE(text_input)
    if not processed_text.strip(): 
        return {"label": CONFIG['labels'][0], "confidence": 0.0} # Default to not_spam

    vectorized_text = VECTORIZER.transform([processed_text])
    probabilities = MODEL.predict_proba(vectorized_text)[0]
    spam_probability = probabilities[1] 

    
    if spam_probability >= CONFIG['confidence_threshold']:
        predicted_label_api = CONFIG['labels'][1] 
        confidence_score_api = float(spam_probability)
    else:
        predicted_label_api = CONFIG['labels'][0] 
        confidence_score_api = float(1.0 - spam_probability) 

    if CONFIG.get('log_predictions', False):
        print(f"Input: '{text_input[:50]}...' -> Processed: '{processed_text[:50]}...' -> Predicted: {predicted_label_api}, Confidence: {confidence_score_api:.4f} (Spam Prob: {spam_probability:.4f})")

    return {
        "label": predicted_label_api,
        "confidence": round(confidence_score_api, 2) 
    }

if __name__ == "__main__":
    import json
    mock_config_path = '../config.json' 
    if not os.path.exists(mock_config_path):
         mock_config_path = 'config.json' 
    try:
        with open(mock_config_path, 'r') as f:
            test_config = json.load(f)
        load_model_and_vectorizer(test_config)
        
        test_texts = [
            "Incase Notice koi na dekha ho aj evry clas md jake ekbar announc kar dena.",
            "zyada itrao mat . english meko bi aati hai :-P ",
            "URGENT call this number for 10000 INR prize",
            "Unlimited internet surf karein (200 MB tak) poore 3 din ke liye sirf Rs18 mein!"
        ]
        for text in test_texts:
            result = predict_spam(text)
            print(f"'{text}' -> {result}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure config.json and model files are present and paths are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")