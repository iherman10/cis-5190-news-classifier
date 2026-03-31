# =============================================================================
# 0. IMPORTS
# =============================================================================

import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# 1. DATA COLLECTION
# =============================================================================

# --- Baseline Model ---
base_url_df = pd.read_csv("data/url_only_data.csv")

if os.path.exists("data/base_scraped_headlines.csv"):
    news_df = pd.read_csv("data/base_scraped_headlines.csv")
else:
    headlines = []
    for i, url in enumerate(base_url_df["url"]):
        print(f"Scraping {i + 1}/{len(base_url_df)}: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            headlines.append({"headline": None, "source": None})
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        if "foxnews.com" in url:
            title = soup.find("h1", class_="headline speakable")
            source = "FoxNews"
        else:
            title = soup.find("h1")
            source = "NBC"
        headlines.append(
            {"headline": title.get_text() if title else None, "source": source}
        )
    news_df = pd.DataFrame(headlines)
    news_df.to_csv("data/base_scraped_headlines.csv", index=False)


# =============================================================================
# 2. DATA SPLITTING
# =============================================================================

# (80% train, 20% test)
# ... ...


# =============================================================================
# 3. DATA CLEANING & PREPROCESSING
# =============================================================================

# --- Baseline Model ---
y_train = y_train.apply(lambda x: 1 if x == "FoxNews" else 0)
y_test = y_test.apply(lambda x: 1 if x == "FoxNews" else 0)

vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =============================================================================
# 4. MODEL TRAINING
# =============================================================================

# --- Baseline Model ---
model = LogisticRegression(max_iter=100)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)


# =============================================================================
# 5. MODEL EVALUATION
# =============================================================================

# --- Baseline Model ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
