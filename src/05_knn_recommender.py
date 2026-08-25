import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. LOAD DATASET
# ============================================================

games = pd.read_csv("dataset/games.csv", index_col=False)

print("Dataset shape:", games.shape)
print("Number of games:", len(games))


# ============================================================
# 2. BASIC CLEANING
# ============================================================

games["AppID"] = games["AppID"].astype(str)

games["Name"] = (
    games["Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove unnecessary column
if "Movies" in games.columns:
    games = games.drop(columns=["Movies"])


# ============================================================
# 3. CLEAN TEXT FEATURES
# ============================================================

features = [
    "About the game",
    "Genres",
    "Tags",
    "Categories"
]

for column in features:

    if column in games.columns:

        games[column] = (
            games[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        # Remove placeholder values
        games[column] = games[column].replace(
            ["0", "0.0", "nan", "None"],
            "",
            regex=False
        )


# ============================================================
# 4. CREATE COMBINED FEATURES
# ============================================================

games["combined_features"] = (
    games["Name"] + " " +
    games["Name"] + " " +
    games["Genres"] + " " +
    games["Tags"] + " " +
    games["Categories"] + " " +
    games["About the game"]
)


# ============================================================
# 5. REMOVE EMPTY FEATURES
# ============================================================

games["combined_features"] = (
    games["combined_features"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# ============================================================
# 6. SHOW FEATURE EXAMPLES
# ============================================================

print("\nSample combined features:")

print(
    games[["Name", "combined_features"]]
    .head(5)
    .to_string(index=False)
)


# ============================================================
# 7. TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    games["combined_features"]
)

print("\nTF-IDF matrix shape:", tfidf_matrix.shape)


# ============================================================
# 8. KNN MODEL
# ============================================================

knn = NearestNeighbors(
    n_neighbors=6,
    metric="cosine",
    algorithm="brute"
)

knn.fit(tfidf_matrix)


# ============================================================
# 9. RECOMMENDATION FUNCTION
# ============================================================

def recommend_game(game_name):

    matches = games[
        games["Name"].str.lower() == game_name.lower()
    ]

    if matches.empty:
        print("\nGame not found.")
        return

    game_index = matches.index[0]

    # Check whether the selected game has useful information
    feature_text = games.loc[
        game_index, "combined_features"
    ]

    if len(feature_text.strip()) < 10:

        print("\nWarning:")
        print("This game has very little metadata.")
        print("Try another game with more information.")
        return

    distances, indices = knn.kneighbors(
        tfidf_matrix[game_index],
        n_neighbors=6
    )

    print("\nSelected Game:")
    print(games.loc[game_index, "Name"])

    print("\nRecommended Games:")
    print("------------------")

    for i in range(1, len(indices[0])):

        recommended_index = indices[0][i]

        similarity = 1 - distances[0][i]

        print(
            f"{i}. "
            f"{games.iloc[recommended_index]['Name']} "
            f"(Similarity: {similarity:.4f})"
        )


# ============================================================
# 10. DISPLAY FIRST FEW GAMES
# ============================================================

print("\nFirst 5 games:")

print(
    games[
        ["AppID", "Name", "Release date"]
    ].head(5)
)


# ============================================================
# 11. FIND A GAME WITH GOOD METADATA FOR TESTING
# ============================================================

games_with_features = games[
    games["combined_features"].str.len() > 100
]

print("\nGames with useful metadata:",
      len(games_with_features))


if len(games_with_features) > 0:

    test_game = games_with_features.iloc[0]["Name"]

    print("\nTest game:")
    print(test_game)

    recommend_game(test_game)

else:

    print("\nNo suitable test game found.")