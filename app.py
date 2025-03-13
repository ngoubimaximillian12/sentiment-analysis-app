import re
import joblib
import logging
import nltk
import ssl
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

# SSL Fix for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure NLTK
NLTK_DIR = "/Users/ngoubimaximilliandiamgha/nltk_data"
nltk.data.path.append(NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)
nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download("wordnet", download_dir=NLTK_DIR)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load the trained sentiment analysis model and vectorizer
try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    logger.info("✅ Model & Vectorizer Loaded Successfully!")
except FileNotFoundError as e:
    logger.error(f"❌ Model/vectorizer file not found: {e}")
    raise

# Connect to the database
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

# Initialize FastAPI application
app = FastAPI()

# Define the Review request schema
class ReviewRequest(BaseModel):
    review: str

# Text preprocessing and cleaning function
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
        # Clean and predict sentiment
        cleaned_review = clean_text(request.review)
        vect_review = vectorizer.transform([cleaned_review])
        prediction = model.predict(vect_review)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

        # Store prediction in database
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO review_sentiment (review, sentiment)
                VALUES (%s, %s);
                """,
                (request.review, sentiment)
            )
            conn.commit()

        return {
            "sentiment": sentiment,
            "review": request.review
        }
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Close the database connection on shutdown
@app.on_event("shutdown")
def shutdown():
    if conn:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
