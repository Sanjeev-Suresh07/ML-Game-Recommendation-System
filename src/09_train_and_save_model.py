import pandas as pd
import numpy as np
import csv
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 1. LOAD DATASET WITH CORRECTED HEADER
# ============================================================

print("Loading dataset...")

csv_path = "dataset/games.csv"


with open(
    csv_path,
    "r",
    encoding="utf-8-sig",
    errors="replace",
    newline=""
) as file:

    reader = csv.reader(file)

    header = next(reader)

    rows = []

    for row in reader:

        if len(row) == 40:

            rows.append(row)

        elif len(row) == 39:

            # Keep normally structured rows
            rows.append(row)

        else:

            # Skip unexpected rows
            continue


# ============================================================
# 2. FIX THE BROKEN HEADER
# ============================================================

# Original header contains:
#
# Price
# DiscountDLC count
# About the game
#
# But actual rows contain:
#
# Price
# Discount
# DLC count
# About the game


fixed_header = []

for column in header:

    if column == "DiscountDLC count":

        fixed_header.append("Discount")
        fixed_header.append("DLC count")

    else:

        fixed_header.append(column)


print(
    "Original header columns:",
    len(header)
)

print(
    "Corrected header columns:",
    len(fixed_header)
)


# ============================================================
# 3. ALIGN DATA WITH CORRECTED HEADER
# ============================================================

corrected_rows = []

for row in rows:

    if len(row) == 40:

        corrected_rows.append(row)

    elif len(row) == 39:

        # This should rarely happen.
        # Insert an empty DLC count field.

        row = (
            row[:8]
            +
            [""]
            +
            row[8:]
        )

        corrected_rows.append(row)


games = pd.DataFrame(
    corrected_rows,
    columns=fixed_header
)


print(
    "\nOriginal dataset:",
    games.shape
)


# ============================================================
# 4. CLEAN GAME NAMES
# ============================================================

games["Name"] = (
    games["Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# 5. REMOVE EMPTY NAMES
# ============================================================

games = games[
    games["Name"] != ""
].copy()


# ============================================================
# 6. REMOVE DUPLICATES
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
# 7. CLEAN TEXT FEATURES
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
# 8. CREATE COMBINED FEATURES
# ============================================================

games["combined_features"] = (

    games["Name"] + " "

    + games["Genres"] + " "

    + games["Genres"] + " "

    + games["Tags"] + " "

    + games["Tags"] + " "

    + games["Categories"] + " "

    + games["About the game"]

)


# ============================================================
# 9. CREATE TF-IDF MODEL
# ============================================================

print(
    "\nCreating TF-IDF model..."
)


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
# 10. PREPARE REVIEW DATA
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
# 11. VERIFY IMPORTANT GAMES
# ============================================================

print(
    "\n========================================"
)

print(
    "REVIEW DATA VERIFICATION"
)

print(
    "========================================"
)


counter_strike = games[
    games["Name"].str.lower()
    ==
    "counter-strike 2"
]


if not counter_strike.empty:

    print(
        "\nCounter-Strike 2:"
    )

    print(
        counter_strike[
            [
                "Name",
                "Positive",
                "Negative"
            ]
        ].to_string(
            index=False
        )
    )


pubg = games[
    games["Name"].str.lower()
    ==
    "pubg: battlegrounds"
]


if not pubg.empty:

    print(
        "\nPUBG: BATTLEGROUNDS:"
    )

    print(
        pubg[
            [
                "Name",
                "Positive",
                "Negative"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 12. REVIEW FEATURES
# ============================================================

total_reviews = (

    games["Positive"]

    +

    games["Negative"]

)


games["has_reviews"] = (
    total_reviews > 0
)


games["positive_ratio"] = np.where(

    total_reviews > 0,

    games["Positive"] /
    total_reviews,

    0

)


# ============================================================
# 13. REVIEW STRENGTH
# ============================================================

games["review_strength"] = np.log1p(
    total_reviews
)


max_strength = games[
    "review_strength"
].max()


if max_strength > 0:

    games["review_strength"] = (

        games["review_strength"]
        /
        max_strength

    )


# ============================================================
# 14. BASIC QUALITY SCORE
# ============================================================

games["quality_score"] = (

    games["positive_ratio"]

    *

    games["review_strength"]

)


# ============================================================
# 15. NORMALIZE QUALITY SCORE
# ============================================================

reviewed = games[
    "has_reviews"
]


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
            ]

            /

            max_quality

        )


# ============================================================
# 16. DATA CHECK
# ============================================================

print(
    "\n========================================"
)

print(
    "DATA CHECK"
)

print(
    "========================================"
)


print(
    "Games with reviews:",
    int(
        games["has_reviews"].sum()
    )
)


print(
    "Total positive reviews:",
    int(
        games["Positive"].sum()
    )
)


print(
    "Total negative reviews:",
    int(
        games["Negative"].sum()
    )
)


# ============================================================
# 17. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)


print(
    "\nSaved: models/tfidf_vectorizer.pkl"
)


# ============================================================
# 18. SAVE TF-IDF MATRIX
# ============================================================

joblib.dump(
    tfidf_matrix,
    "models/tfidf_matrix.pkl"
)


print(
    "Saved: models/tfidf_matrix.pkl"
)


# ============================================================
# 19. SAVE GAME DATA
# ============================================================

games.to_pickle(
    "models/games_data.pkl"
)


print(
    "Saved: models/games_data.pkl"
)


# ============================================================
# 20. FINAL
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


print(
    "Games:",
    len(games)
)

print(
    "TF-IDF matrix:",
    tfidf_matrix.shape
)