import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 1. LOAD DATASET
# ============================================================

games = pd.read_csv(
    "dataset/games.csv",
    index_col=False,
    low_memory=False
)

print("Original dataset:", games.shape)


# ============================================================
# 2. REMOVE DUPLICATE GAMES
# ============================================================

games["Name"] = (
    games["Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

games = games.drop_duplicates(
    subset=["Name"],
    keep="first"
).reset_index(drop=True)

print("After removing duplicates:", games.shape)


# ============================================================
# 3. CLEAN TEXT FEATURES
# ============================================================

text_columns = [
    "About the game",
    "Genres",
    "Tags",
    "Categories"
]

for column in text_columns:

    games[column] = (
        games[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# 4. CREATE COMBINED FEATURES
# ============================================================

games["combined_features"] = (
    games["Name"] + " " +
    games["Genres"] + " " +
    games["Genres"] + " " +
    games["Tags"] + " " +
    games["Tags"] + " " +
    games["Categories"] + " " +
    games["About the game"]
)


# ============================================================
# 5. CREATE TF-IDF MODEL
# ============================================================

print("\nCreating TF-IDF model...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    games["combined_features"]
)

print(
    "TF-IDF matrix:",
    tfidf_matrix.shape
)


# ============================================================
# 6. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print(
    "Saved: models/tfidf_vectorizer.pkl"
)


# ============================================================
# 7. SAVE TF-IDF MATRIX
# ============================================================

joblib.dump(
    tfidf_matrix,
    "models/tfidf_matrix.pkl"
)

print(
    "Saved: models/tfidf_matrix.pkl"
)


# ============================================================
# 8. SAVE GAME DATA
# ============================================================

games.to_pickle(
    "models/games_data.pkl"
)

print(
    "Saved: models/games_data.pkl"
)


# ============================================================
# 9. TRAINING COMPLETE
# ============================================================

print(
    "\n========================================"
)

print(
    "MODEL TRAINING AND SAVING COMPLETE"
)

print(
    "========================================"
)

print(
    "\nSaved files:"
)

print(
    "1. models/tfidf_vectorizer.pkl"
)

print(
    "2. models/tfidf_matrix.pkl"
)

print(
    "3. models/games_data.pkl"
)