import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os


try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.word_tokenize("test")
except LookupError:
    nltk.download('punkt')


DATASET_PATH = 'data/dataset.csv'
MODEL_DIR = 'app/model'
MODEL_CHOICE = 'naive_bayes' 
# MODEL_CHOICE = 'logistic_regression'


os.makedirs(MODEL_DIR, exist_ok=True)

stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'\@\w+|\#','', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    text_tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in text_tokens if word not in stop_words and word.isalpha()]
    return " ".join(filtered_text)

print(f"Loading dataset from {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded. Shape: {df.shape}")
print(df.head())

if df.empty:
    print("Dataset is empty. Exiting.")
    exit()

print("Preprocessing text data...")
df['processed_text'] = df['v2'].astype(str).apply(preprocess_text)
print(df[['v2', 'processed_text']].head())

print("Vectorizing text using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=3000) 
X = tfidf_vectorizer.fit_transform(df['processed_text'])
y = df['v1'].apply(lambda x: 1 if x == 'spam' else 0) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")


if MODEL_CHOICE == 'naive_bayes':
    print("Training Naive Bayes model...")
    model = MultinomialNB()
elif MODEL_CHOICE == 'logistic_regression':
    print("Training Logistic Regression model...")
    model = LogisticRegression(solver='liblinear', random_state=42) 
else:
    raise ValueError("Invalid MODEL_CHOICE. Choose 'naive_bayes' or 'logistic_regression'.")

model.fit(X_train, y_train)
print("Model training complete.")
print("\n--- Model Evaluation ---")
y_pred_train = model.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")

y_pred_test = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))
print("\nTest Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=['not_spam', 'spam']))

MODEL_PATH = os.path.join(MODEL_DIR, f'{MODEL_CHOICE}_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

print(f"\nSaving model to {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)
print(f"Saving vectorizer to {VECTORIZER_PATH}")
joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)

print("\n--- Training and saving complete! ---")
print(f"Model artifacts saved in '{MODEL_DIR}' directory.")


# loaded_model = joblib.load(MODEL_PATH)
# loaded_vectorizer = joblib.load(VECTORIZER_PATH)
# sample_text = "free money click now"
# processed_sample = preprocess_text(sample_text)
# sample_vec = loaded_vectorizer.transform([processed_sample])
# prediction = loaded_model.predict(sample_vec)
# probability = loaded_model.predict_proba(sample_vec)
# print(f"Sample prediction for '{sample_text}': {prediction[0]}, Proba: {probability[0]}")