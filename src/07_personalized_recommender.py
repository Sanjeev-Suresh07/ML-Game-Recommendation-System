import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. LOAD DATASET
# ============================================================

games = pd.read_csv(
    "dataset/games.csv",
    index_col=False,
    low_memory=False
)

print("Dataset shape:", games.shape)


# ============================================================
# 2. CHECK IMPORTANT COLUMNS
# ============================================================

print("\nFirst 5 games:")
print(
    games[["AppID", "Name", "Release date"]]
    .head()
    .to_string(index=False)
)


# ============================================================
# 3. CLEAN DATA
# ============================================================

games["AppID"] = games["AppID"].astype(str)

games["Name"] = (
    games["Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

features = [
    "About the game",
    "Genres",
    "Tags",
    "Categories"
]

for column in features:
    games[column] = (
        games[column]
        .fillna("")
        .astype(str)
    )


# ============================================================
# 4. CREATE COMBINED GAME FEATURES
# ============================================================

games["combined_features"] = (
    games["About the game"] + " " +
    games["Genres"] + " " +
    games["Tags"] + " " +
    games["Categories"]
)

# Remove games without useful information
games = games[
    games["combined_features"].str.strip() != ""
].reset_index(drop=True)

print("\nGames with useful metadata:", len(games))


# ============================================================
# 5. TF-IDF FEATURE ENGINEERING
# ============================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = tfidf.fit_transform(
    games["combined_features"]
)

print(
    "TF-IDF matrix shape:",
    tfidf_matrix.shape
)


# ============================================================
# 6. CREATE KNN MODEL
# ============================================================

knn = NearestNeighbors(
    n_neighbors=6,
    metric="cosine",
    algorithm="brute"
)

knn.fit(tfidf_matrix)


# ============================================================
# 7. USER PROFILE
# ============================================================

user_name = "User"

played_games = [
    "Counter-Strike 2",
    "PUBG: BATTLEGROUNDS"
]

preferred_genres = [
    "Action",
    "Shooter",
    "Multiplayer"
]

preferred_tags = [
    "FPS",
    "Competitive",
    "Online Multiplayer"
]


# ============================================================
# 8. DISPLAY USER PROFILE
# ============================================================

print("\n================ USER PROFILE ================")

print("User:", user_name)

print("\nGames Played:")

for game in played_games:
    print("-", game)

print("\nPreferred Genres:")

for genre in preferred_genres:
    print("-", genre)

print("\nPreferred Tags:")

for tag in preferred_tags:
    print("-", tag)


# ============================================================
# 9. NORMALIZE GAME NAMES
# ============================================================

def normalize_name(name):
    return (
        str(name)
        .lower()
        .replace(":", "")
        .replace("-", " ")
        .replace("_", " ")
        .strip()
    )


games["normalized_name"] = games["Name"].apply(
    normalize_name
)


# ============================================================
# 10. FIND PLAYED GAMES
# ============================================================

matched_indices = []

print("\n================ GAME CHECK ================")

for played_game in played_games:

    target = normalize_name(played_game)

    # First try exact matching
    matches = games[
        games["normalized_name"] == target
    ]

    # If exact match fails, try partial matching
    if matches.empty:

        matches = games[
            games["normalized_name"].str.contains(
                target,
                regex=False,
                na=False
            )
        ]

    # Try matching important words if still not found
    if matches.empty:

        words = target.split()

        if len(words) >= 2:

            matches = games[
                games["normalized_name"].apply(
                    lambda x: all(word in x for word in words)
                )
            ]

    if not matches.empty:

        index = matches.index[0]

        matched_indices.append(index)

        print(
            "Found:",
            played_game,
            "->",
            games.loc[index, "Name"]
        )

    else:

        print(
            "Not found:",
            played_game
        )


print(
    "\nNumber of matched games:",
    len(matched_indices)
)


# ============================================================
# 11. STOP IF NO GAMES WERE FOUND
# ============================================================

if len(matched_indices) == 0:

    print(
        "\nNo played games were found in the dataset."
    )

    print(
        "Cannot create personalized recommendations."
    )

else:

    # ========================================================
    # 12. CREATE USER PROFILE VECTOR
    # ========================================================

    played_vectors = tfidf_matrix[
        matched_indices
    ]

    # Average the features of played games
    user_vector = played_vectors.mean(
        axis=0
    )

    user_vector = np.asarray(
        user_vector
    ).reshape(1, -1)


    # ========================================================
    # 13. ADD USER PREFERENCES
    # ========================================================

    preference_text = (
        " ".join(preferred_genres) + " " +
        " ".join(preferred_tags)
    )

    preference_vector = tfidf.transform(
        [preference_text]
    )


    # Combine played-game profile and preferences
    user_profile = (
        user_vector * 0.7 +
        preference_vector.toarray() * 0.3
    )


    # ========================================================
    # 14. CREATE PERSONALIZED KNN SEARCH
    # ========================================================

    distances, indices = knn.kneighbors(
        user_profile,
        n_neighbors=20
    )


    # ========================================================
    # 15. DISPLAY RECOMMENDATIONS
    # ========================================================

    print(
        "\n================ PERSONALIZED RECOMMENDATIONS ================"
    )

    recommendation_count = 0

    for i in range(len(indices[0])):

        index = indices[0][i]

        # Don't recommend already played games
        if index in matched_indices:
            continue

        game_name = games.loc[
            index,
            "Name"
        ]

        distance = distances[0][i]

        similarity = 1 - distance

        recommendation_count += 1

        print(
            f"{recommendation_count}. "
            f"{game_name} "
            f"(Similarity: {similarity:.4f})"
        )

        if recommendation_count == 5:
            break


    # ========================================================
    # 16. PROFILE SUMMARY
    # ========================================================

    print(
        "\n================ PROFILE SUMMARY ================"
    )

    print(
        "User:",
        user_name
    )

    print(
        "Played games:",
        len(matched_indices)
    )

    print(
        "Preferred genres:",
        ", ".join(preferred_genres)
    )

    print(
        "Preferred tags:",
        ", ".join(preferred_tags)
    )