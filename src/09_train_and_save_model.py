import pandas as pd
import numpy as np
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
# 2. CLEAN GAME NAMES
# ============================================================

games["Name"] = (
    games["Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

games = games.drop_duplicates(
    subset=["Name"],
    keep="first"
).reset_index(drop=True)

print(
    "After removing duplicates:",
    games.shape
)


# ============================================================
# 4. CLEAN TEXT FEATURES
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
# 5. CREATE COMBINED FEATURES
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
# 6. CREATE TF-IDF
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
# 7. PREPARE REVIEW DATA
# ============================================================

games["Positive"] = pd.to_numeric(
    games["Positive"],
    errors="coerce"
).fillna(0)

games["Negative"] = pd.to_numeric(
    games["Negative"],
    errors="coerce"
).fillna(0)


# ============================================================
# 8. CALCULATE REVIEW QUALITY
# ============================================================

total_reviews = (
    games["Positive"] +
    games["Negative"]
)

games["has_reviews"] = (
    total_reviews > 0
)

games["positive_ratio"] = np.where(
    total_reviews > 0,
    games["Positive"] / total_reviews,
    0
)


# Review strength
games["review_strength"] = np.log1p(
    total_reviews
)

max_strength = games["review_strength"].max()

if max_strength > 0:

    games["review_strength"] = (
        games["review_strength"] /
        max_strength
    )


# Final quality score
games["quality_score"] = (
    games["positive_ratio"] *
    games["review_strength"]
)


# ============================================================
# 9. NORMALIZE QUALITY SCORE
# ============================================================

reviewed = games["has_reviews"]

if reviewed.any():

    max_quality = games.loc[
        reviewed,
        "quality_score"
    ].max()

    if max_quality > 0:

        games.loc[
            reviewed,
            "quality_score"
        ] = (
            games.loc[
                reviewed,
                "quality_score"
            ] / max_quality
        )


# ============================================================
# 10. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print(
    "Saved: models/tfidf_vectorizer.pkl"
)


# ============================================================
# 11. SAVE TF-IDF MATRIX
# ============================================================

joblib.dump(
    tfidf_matrix,
    "models/tfidf_matrix.pkl"
)

print(
    "Saved: models/tfidf_matrix.pkl"
)


# ============================================================
# 12. SAVE COMPLETE GAME DATA
# ============================================================

games.to_pickle(
    "models/games_data.pkl"
)

print(
    "Saved: models/games_data.pkl"
)


# ============================================================
# 13. VERIFY IMPORTANT COLUMNS
# ============================================================

print(
    "\nSaved game data columns:"
)

print(
    "quality_score:",
    "quality_score" in games.columns
)

print(
    "has_reviews:",
    "has_reviews" in games.columns
)


# ============================================================
# 14. COMPLETE
# ============================================================

print(
    "\n========================================"
)

print(
    "FINAL MODEL DATA SAVED SUCCESSFULLY"
)

print(
    "========================================"
)