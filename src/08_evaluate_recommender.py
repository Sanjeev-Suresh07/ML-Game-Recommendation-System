import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD DATASET
# ============================================================

games = pd.read_csv(
    "dataset/games.csv",
    index_col=False,
    low_memory=False
)

games["Name"] = (
    games["Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

print("Dataset shape:", games.shape)


# ============================================================
# 2. REMOVE DUPLICATE GAME NAMES
# ============================================================

before = len(games)

games = games.drop_duplicates(
    subset=["Name"],
    keep="first"
).reset_index(drop=True)

after = len(games)

print("Duplicate game entries removed:", before - after)
print("Unique games:", after)


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
# 5. TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    games["combined_features"]
)

print("TF-IDF matrix shape:", tfidf_matrix.shape)


# ============================================================
# 6. RECOMMENDATION FUNCTION
# ============================================================

def recommend_games(played_games, top_k=5):

    played_indices = []

    # Find played games
    for game_name in played_games:

        matches = games[
            games["Name"].str.lower() ==
            game_name.lower()
        ]

        if not matches.empty:

            played_indices.append(
                matches.index[0]
            )


    if len(played_indices) == 0:

        return [], 0


    # Create user profile
    user_vector = tfidf_matrix[
        played_indices
    ].mean(axis=0)

    user_vector = np.asarray(
        user_vector
    )


    # Calculate similarity
    scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()


    # Remove played games
    scores[played_indices] = -1


    # Get more candidates first
    candidate_count = min(
        top_k * 3,
        len(games)
    )

    candidate_indices = np.argsort(
        scores
    )[::-1][:candidate_count]


    # Remove duplicate names
    recommendations = []
    used_names = set()

    for index in candidate_indices:

        game_name = games.loc[
            index,
            "Name"
        ]

        name_key = game_name.lower().strip()

        if name_key in used_names:
            continue

        used_names.add(name_key)

        recommendations.append(
            game_name
        )

        if len(recommendations) == top_k:
            break


    return recommendations, len(played_indices)


# ============================================================
# 7. TEST PROFILES
# ============================================================

profile_1 = [
    "Counter-Strike 2",
    "PUBG: BATTLEGROUNDS"
]

profile_2 = [
    "MORDHAU",
    "War Thunder"
]

profile_3 = [
    "Team Fortress 2",
    "Left 4 Dead 2"
]


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate_profile(
    profile_name,
    played_games
):

    recommendations, matched = recommend_games(
        played_games
    )

    print(
        "\n================================================"
    )

    print(profile_name)

    print(
        "================================================"
    )

    print("\nPlayed games:")

    for game in played_games:
        print("-", game)

    print(
        "\nMatched games:",
        matched
    )

    print("\nTop recommendations:")

    for number, game in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{number}. {game}"
        )

    return recommendations


# ============================================================
# 9. RUN TESTS
# ============================================================

recommendations_1 = evaluate_profile(
    "PROFILE 1 - FPS / ACTION",
    profile_1
)

recommendations_2 = evaluate_profile(
    "PROFILE 2 - ACTION / MULTIPLAYER",
    profile_2
)

recommendations_3 = evaluate_profile(
    "PROFILE 3 - MULTIPLAYER / ACTION",
    profile_3
)


# ============================================================
# 10. RELEVANCE KEYWORDS
# ============================================================

relevant_keywords = [
    "action",
    "shooter",
    "multiplayer",
    "fps",
    "online",
    "competitive"
]


# ============================================================
# 11. CALCULATE RELEVANCE
# ============================================================

def calculate_relevance(
    recommendations
):

    if len(recommendations) == 0:
        return 0

    relevant_count = 0

    for game in recommendations:

        game_row = games[
            games["Name"] == game
        ]

        if game_row.empty:
            continue

        row = game_row.iloc[0]

        text = (
            str(row["Genres"]) + " " +
            str(row["Tags"]) + " " +
            str(row["Categories"])
        ).lower()

        for keyword in relevant_keywords:

            if keyword in text:

                relevant_count += 1
                break


    return (
        relevant_count /
        len(recommendations)
    )


# ============================================================
# 12. PRECISION@5
# ============================================================

def precision_at_5(
    recommendations
):

    if len(recommendations) == 0:
        return 0

    relevant = 0

    for game in recommendations:

        game_row = games[
            games["Name"] == game
        ]

        if game_row.empty:
            continue

        row = game_row.iloc[0]

        text = (
            str(row["Genres"]) + " " +
            str(row["Tags"]) + " " +
            str(row["Categories"])
        ).lower()

        is_relevant = any(
            keyword in text
            for keyword in relevant_keywords
        )

        if is_relevant:
            relevant += 1


    return relevant / 5


# ============================================================
# 13. EVALUATION RESULTS
# ============================================================

print(
    "\n================================================"
)

print("EVALUATION RESULTS")

print(
    "================================================"
)


precision_1 = precision_at_5(
    recommendations_1
)

precision_2 = precision_at_5(
    recommendations_2
)

precision_3 = precision_at_5(
    recommendations_3
)


print(
    "\nProfile 1 Precision@5:",
    f"{precision_1:.2f}"
)

print(
    "Profile 2 Precision@5:",
    f"{precision_2:.2f}"
)

print(
    "Profile 3 Precision@5:",
    f"{precision_3:.2f}"
)


average_precision = (
    precision_1 +
    precision_2 +
    precision_3
) / 3


print(
    "\nAverage Precision@5:",
    f"{average_precision:.2f}"
)


print(
    "\nEvaluation complete."
)