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
# 2. CLEAN TEXT FEATURES
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
# 3. CREATE COMBINED FEATURES
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
# 4. CREATE TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    games["combined_features"]
)

print(
    "TF-IDF matrix shape:",
    tfidf_matrix.shape
)


# ============================================================
# 5. RECOMMENDATION FUNCTION
# ============================================================

def recommend_games(played_games, top_k=5):

    # --------------------------------------------------------
    # Find games in dataset
    # --------------------------------------------------------

    played_indices = []

    for game_name in played_games:

        matches = games[
            games["Name"].str.lower() ==
            game_name.lower()
        ]

        if not matches.empty:

            played_indices.append(
                matches.index[0]
            )


    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    if len(played_indices) == 0:

        return [], 0


    # --------------------------------------------------------
    # Create user vector
    # --------------------------------------------------------

    user_vector = tfidf_matrix[
        played_indices
    ].mean(axis=0)

    user_vector = np.asarray(
        user_vector
    )


    # --------------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------------

    scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()


    # --------------------------------------------------------
    # Remove already played games
    # --------------------------------------------------------

    scores[played_indices] = -1


    # --------------------------------------------------------
    # Get recommendations
    # --------------------------------------------------------

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]


    recommendations = []

    for index in top_indices:

        recommendations.append(
            games.loc[index, "Name"]
        )


    return recommendations, len(played_indices)


# ============================================================
# 6. TEST PROFILE 1 — FPS / ACTION
# ============================================================

profile_1 = [
    "Counter-Strike 2",
    "PUBG: BATTLEGROUNDS"
]


# ============================================================
# 7. TEST PROFILE 2 — USE ACTUAL DATASET GAMES
# ============================================================

profile_2 = [
    "MORDHAU",
    "War Thunder"
]


# ============================================================
# 8. TEST PROFILE 3
# ============================================================

profile_3 = [
    "Team Fortress 2",
    "Left 4 Dead 2"
]


# ============================================================
# 9. EVALUATION FUNCTION
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

    print(
        profile_name
    )

    print(
        "================================================"
    )

    print(
        "\nPlayed games:"
    )

    for game in played_games:
        print(
            "-",
            game
        )

    print(
        "\nMatched games:",
        matched
    )

    print(
        "\nTop recommendations:"
    )

    for number, game in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{number}. {game}"
        )


    return recommendations


# ============================================================
# 10. RUN TESTS
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
# 11. SIMPLE RELEVANCE CHECK
# ============================================================

print(
    "\n================================================"
)

print(
    "SIMPLE EVALUATION"
)

print(
    "================================================"
)


# Keywords that represent relevant games
relevant_keywords = [
    "action",
    "shooter",
    "multiplayer",
    "fps",
    "online",
    "competitive"
]


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

        found = False

        for keyword in relevant_keywords:

            if keyword in text:

                found = True
                break

        if found:

            relevant_count += 1


    return (
        relevant_count /
        len(recommendations)
    )


# ============================================================
# 12. CALCULATE RESULTS
# ============================================================

score_1 = calculate_relevance(
    recommendations_1
)

score_2 = calculate_relevance(
    recommendations_2
)

score_3 = calculate_relevance(
    recommendations_3
)


print(
    "\nProfile 1 relevance:",
    f"{score_1:.2f}"
)

print(
    "Profile 2 relevance:",
    f"{score_2:.2f}"
)

print(
    "Profile 3 relevance:",
    f"{score_3:.2f}"
)


average_score = (
    score_1 +
    score_2 +
    score_3
) / 3


print(
    "\nAverage relevance:",
    f"{average_score:.2f}"
)


print(
    "\nEvaluation complete."
)