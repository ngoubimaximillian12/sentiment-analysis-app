import re
import nltk
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

NLTK_DIR = "/Users/ngoubimaximilliandiamgha/nltk_data"
nltk.data.path.append(NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)
nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download("wordnet", download_dir=NLTK_DIR)

stop_words = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(tokens)

df = pd.read_csv("data/IMDBDataset.csv")
df["clean_review"] = df["review"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"].map({"positive": 1, "negative": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("✅ Model & Vectorizer Saved Successfully!")
