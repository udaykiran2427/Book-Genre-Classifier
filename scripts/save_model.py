"""
Train the genre classification model and save all artifacts.
"""

import pandas as pd
import numpy as np
import ast
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report

# file paths
CSV_PATH = "data/raw/books_1.Best_Books_Ever.csv"
CLEANED_OUT = "data/processed/cleaned_books.csv"
MODEL_OUT = "model_store/genre_model.pkl"
TFIDF_OUT = "model_store/tfidf.pkl"
MLB_OUT = "model_store/mlb.pkl"

# final genres
TARGET_GENRES = ['Fantasy', 'Nonfiction', 'Romance', 'Young Adult']

# keyword mapping
GENRE_KEYWORDS = {
    'Fantasy': ['fantasy', 'epic fantasy', 'high fantasy', 'urban fantasy', 'magic', 'dragon'],
    'Nonfiction': ['nonfiction', 'memoir', 'biography', 'history', 'true story'],
    'Romance': ['romance', 'romantic', 'love story'],
    'Young Adult': ['young adult', 'ya', 'teen', 'coming of age']
}

# parse genre strings into lists
def safe_literal_eval(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('[') and val.endswith(']'):
            try:
                return ast.literal_eval(val)
            except:
                pass
        return [val]
    return []

# map messy genres to our 4 main ones
def recategorize_fuzzy(genre_list):
    found = set()
    for g in genre_list:
        g_low = str(g).lower()
        for tgt, keywords in GENRE_KEYWORDS.items():
            if any(kw in g_low for kw in keywords):
                found.add(tgt)
    return list(found) if found else ['Other']

# custom metric
def partial_match_accuracy(y_true, y_pred):
    matches = 0
    for true_row, pred_row in zip(y_true, y_pred):
        true_idx = np.where(true_row == 1)[0]
        pred_idx = np.where(pred_row == 1)[0]
        if len(np.intersect1d(true_idx, pred_idx)) > 0:
            matches += 1
    return matches / len(y_true)


# load dataset
print("Loading dataset...")
df = pd.read_csv(CSV_PATH, engine='python', on_bad_lines='skip')

# clean descriptions
df['description'] = df['description'].replace(r'^\s*$', np.nan, regex=True)
df.dropna(subset=['description'], inplace=True)
df['description'] = df['description'].astype(str)

# parse and remap genres
df['genres'] = df['genres'].apply(safe_literal_eval)
df['final_genres'] = df['genres'].apply(recategorize_fuzzy)

# drop books with no matching target genre
df = df[df['final_genres'].apply(lambda x: x != ['Other'])]

print("Dataset size after cleaning:", len(df))

# save cleaned dataset
os.makedirs("data/processed", exist_ok=True)
df.to_csv(CLEANED_OUT, index=False)
print("Cleaned dataset saved to", CLEANED_OUT)

# vectorize text
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=3
)
X = tfidf.fit_transform(df['description'])

# encode labels
mlb = MultiLabelBinarizer(classes=TARGET_GENRES)
y = mlb.fit_transform(df['final_genres'])

# split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = OneVsRestClassifier(
    LinearSVC(class_weight='balanced', max_iter=8000)
)

print("Training model...")
model.fit(X_train, y_train)
print("Training complete.")

# evaluate
y_pred = model.predict(X_test)

print("\nPartial Match Accuracy:", partial_match_accuracy(y_test, y_pred))
print("Micro F1:", f1_score(y_test, y_pred, average='micro'))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# save artifacts
joblib.dump(model, MODEL_OUT)
joblib.dump(tfidf, TFIDF_OUT)
joblib.dump(mlb, MLB_OUT)

print("\nModel saved to model_store/")
