import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# =========================================================
# 1. LOAD DATASET
# =========================================================

games = pd.read_csv("dataset/games.csv", index_col=False)

# The CSV contains an extra index column.
# Remove it if present.
if len(games.columns) > 39:
    games = games.iloc[:, 1:]

print("Dataset shape:", games.shape)


# =========================================================
# 2. CHECK / CLEAN COLUMNS
# =========================================================

games["Name"] = games["Name"].fillna("").astype(str).str.strip()

features = [
    "About the game",
    "Genres",
    "Tags",
    "Categories"
]

for column in features:
    games[column] = games[column].fillna("").astype(str)


# =========================================================
# 3. CREATE COMBINED FEATURES
# =========================================================

games["combined_features"] = (
    games["About the game"] + " " +
    games["Genres"] + " " +
    games["Tags"] + " " +
    games["Categories"]
)

print("Number of games:", len(games))


# =========================================================
# 4. TF-IDF
# =========================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(
    games["combined_features"]
)

print("TF-IDF matrix shape:", tfidf_matrix.shape)


# =========================================================
# 5. KNN MODEL
# =========================================================

knn = NearestNeighbors(
    n_neighbors=6,
    metric="cosine",
    algorithm="brute"
)

knn.fit(tfidf_matrix)


# =========================================================
# 6. RECOMMENDATION FUNCTION
# =========================================================

def recommend_game(game_name):

    matches = games[
        games["Name"].str.lower() == game_name.lower()
    ]

    # Try partial search if exact match fails
    if matches.empty:
        matches = games[
            games["Name"].str.lower().str.contains(
                game_name.lower(),
                na=False
            )
        ]

    if matches.empty:
        print("\nGame not found.")
        return

    game_index = matches.index[0]

    print("\nSelected Game:")
    print(games.loc[game_index, "Name"])

    distances, indices = knn.kneighbors(
        tfidf_matrix[game_index]
    )

    print("\nRecommended Games:")
    print("------------------")

    for i in range(1, len(indices[0])):

        recommended_index = indices[0][i]

        similarity = 1 - distances[0][i]

        print(
            f"{i}. "
            f"{games.iloc[recommended_index]['Name']} "
            f"(Similarity: {similarity:.2f})"
        )


# =========================================================
# 7. TEST
# =========================================================

print("\nFirst 3 games:")
print(games[["AppID", "Name", "Release date"]].head(3))

recommend_game("Black Dragon Mage Playtest")