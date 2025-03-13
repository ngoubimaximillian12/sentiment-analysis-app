import re
import joblib
import logging
import nltk
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Database connection details
DB_HOST = "127.0.0.1"
DB_PORT = "5432"
DB_NAME = "sentiment_analysis"
DB_USER = "postgres"
DB_PASSWORD = "hope"

# Connect to the database
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

# Configure NLTK
NLTK_DIR = "/Users/ngoubimaximilliandiamgha/nltk_data"
nltk.data.path.append(NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)
nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download("wordnet", download_dir=NLTK_DIR)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load the trained model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

app = FastAPI()

class ReviewRequest(BaseModel):
    review: str

def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(tokens)

@app.get("/")
def root():
    return {
        "message": "✅ Sentiment Analysis API is running!",
        "predict_endpoint": "http://127.0.0.1:5002/predict"
    }

@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    try:
        cleaned_review = clean_text(request.review)
        vect_review = vectorizer.transform([cleaned_review])
        prediction = model.predict(vect_review)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO review_sentiment (review, sentiment) VALUES (%s, %s);""",
                (request.review, sentiment)
            )
            conn.commit()
        return {"sentiment": sentiment, "review": request.review}
    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
def shutdown():
    if conn:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
