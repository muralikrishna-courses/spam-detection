FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --compile -r requirements.txt
RUN python3 -m nltk.downloader punkt -d /usr/share/nltk_data
RUN python3 -m nltk.downloader stopwords -d /usr/share/nltk_data
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
COPY ./app ./app
COPY ./data ./data
COPY ./config.json ./config.json
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--lifespan", "on"]