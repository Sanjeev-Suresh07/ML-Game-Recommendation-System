import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD SAVED FILES
# ============================================================

print("Loading saved model files...")

tfidf_vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

tfidf_matrix = joblib.load(
    "models/tfidf_matrix.pkl"
)

games = pd.read_pickle(
    "models/games_data.pkl"
)

print("TF-IDF vectorizer loaded")
print("TF-IDF matrix loaded")
print("Game data loaded")


# ============================================================
# 2. DISPLAY MODEL INFORMATION
# ============================================================

print("\n========================================")
print("SAVED MODEL INFORMATION")
print("========================================")

print(
    "Number of games:",
    len(games)
)

print(
    "TF-IDF matrix shape:",
    tfidf_matrix.shape
)


# ============================================================
# 3. RECOMMENDATION FUNCTION
# ============================================================

def recommend_game(game_name, top_k=5):

    matches = games[
        games["Name"].str.lower() ==
        game_name.lower()
    ]

    if matches.empty:

        print("\nGame not found:", game_name)
        return

    game_index = matches.index[0]

    # Calculate similarity
    similarities = cosine_similarity(
        tfidf_matrix[game_index],
        tfidf_matrix
    ).flatten()

    # Remove selected game
    similarities[game_index] = -1

    # Get top games
    top_indices = similarities.argsort()[
        ::-1
    ][:top_k]

    print(
        "\n========================================"
    )

    print(
        "RECOMMENDATIONS"
    )

    print(
        "========================================"
    )

    print(
        "Selected game:",
        games.loc[game_index, "Name"]
    )

    print(
        "\nRecommended games:"
    )

    for number, index in enumerate(
        top_indices,
        start=1
    ):

        print(
            f"{number}. "
            f"{games.loc[index, 'Name']} "
            f"(Similarity: "
            f"{similarities[index]:.4f})"
        )


# ============================================================
# 4. TEST SAVED MODEL
# ============================================================

test_game = "Counter-Strike 2"

recommend_game(
    test_game,
    top_k=5
)


# ============================================================
# 5. COMPLETE
# ============================================================

print(
    "\n========================================"
)

print(
    "SAVED MODEL TEST COMPLETE"
)

print(
    "========================================"
)