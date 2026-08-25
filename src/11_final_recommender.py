import pandas as pd
import numpy as np
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

print("Loading saved recommendation model...")

tfidf_matrix = joblib.load(
    "models/tfidf_matrix.pkl"
)

games = pd.read_pickle(
    "models/games_data.pkl"
)

print("Model loaded successfully.")
print("Games available:", len(games))


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
# 3. RECOMMENDATION FUNCTION
# ============================================================

def recommend_games(
    played_games,
    preferred_genres=None,
    preferred_tags=None,
    top_k=5
):

    # Default preferences
    if preferred_genres is None:
        preferred_genres = []

    if preferred_tags is None:
        preferred_tags = []


    # --------------------------------------------------------
    # Find games played by user
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


    if len(played_indices) == 0:

        print(
            "None of the played games were found."
        )

        return []


    # --------------------------------------------------------
    # Create user profile vector
    # --------------------------------------------------------

    user_vector = tfidf_matrix[
        played_indices
    ].mean(axis=0)

    user_vector = np.asarray(
        user_vector
    )


    # --------------------------------------------------------
    # Content similarity
    # --------------------------------------------------------

    content_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()


    # --------------------------------------------------------
    # Preference matching
    # --------------------------------------------------------

    preferred_genres = [
        x.lower().strip()
        for x in preferred_genres
    ]

    preferred_tags = [
        x.lower().strip()
        for x in preferred_tags
    ]


    preference_scores = []


    for _, row in games.iterrows():

        genres = str(
            row["Genres"]
        ).lower()

        tags = str(
            row["Tags"]
        ).lower()


        # Convert comma-separated data
        game_genres = [
            x.strip()
            for x in genres.split(",")
            if x.strip()
        ]

        game_tags = [
            x.strip()
            for x in tags.split(",")
            if x.strip()
        ]


        # Genre match
        genre_matches = sum(
            genre in game_genres
            for genre in preferred_genres
        )

        if len(preferred_genres) > 0:

            genre_score = (
                genre_matches /
                len(preferred_genres)
            )

        else:

            genre_score = 0


        # Tag match
        tag_matches = sum(
            tag in game_tags
            for tag in preferred_tags
        )

        if len(preferred_tags) > 0:

            tag_score = (
                tag_matches /
                len(preferred_tags)
            )

        else:

            tag_score = 0


        # Combined preference
        preference_score = (
            0.6 * genre_score +
            0.4 * tag_score
        )

        preference_scores.append(
            preference_score
        )


    preference_scores = np.array(
        preference_scores
    )


    # --------------------------------------------------------
    # Quality score
    # --------------------------------------------------------

    if "quality_score" in games.columns:

        quality_scores = (
            games["quality_score"]
            .fillna(0)
            .values
        )

    else:

        quality_scores = np.zeros(
            len(games)
        )


    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    final_scores = (

        0.70 * content_scores +

        0.25 * preference_scores +

        0.05 * quality_scores

    )


    # --------------------------------------------------------
    # Remove already played games
    # --------------------------------------------------------

    final_scores[
        played_indices
    ] = -1


    # --------------------------------------------------------
    # Get more candidates
    # --------------------------------------------------------

    candidate_count = min(
        top_k * 3,
        len(games)
    )

    candidate_indices = np.argsort(
        final_scores
    )[::-1][:candidate_count]


    # --------------------------------------------------------
    # Remove duplicate game names
    # --------------------------------------------------------

    recommendations = []

    used_names = set()


    for index in candidate_indices:

        game_name = games.loc[
            index,
            "Name"
        ]

        name_key = (
            game_name.lower().strip()
        )

        if name_key in used_names:
            continue

        used_names.add(
            name_key
        )


        # ----------------------------------------------------
        # Recommendation explanation
        # ----------------------------------------------------

        reasons = []


        if content_scores[index] >= 0.50:

            reasons.append(
                "similar to your played games"
            )


        if preference_scores[index] > 0:

            reasons.append(
                "matches your preferences"
            )


        if quality_scores[index] > 0:

            reasons.append(
                "has positive review data"
            )


        if not reasons:

            reasons.append(
                "similar game content"
            )


        recommendation = {

            "name": game_name,

            "score": round(
                float(final_scores[index]),
                4
            ),

            "content_similarity": round(
                float(content_scores[index]),
                4
            ),

            "preference_score": round(
                float(preference_scores[index]),
                4
            ),

            "quality_score": round(
                float(quality_scores[index]),
                4
            ),

            "reasons": reasons
        }


        recommendations.append(
            recommendation
        )


        if len(recommendations) == top_k:
            break


    return recommendations


# ============================================================
# 4. TEST THE FINAL RECOMMENDER
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "FINAL PERSONALIZED RECOMMENDER"
    )

    print(
        "========================================"
    )


    # Example user
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


    # Generate recommendations
    recommendations = recommend_games(
        played_games,
        preferred_genres,
        preferred_tags,
        top_k=5
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\nRecommended Games:"
    )


    for number, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"\n{number}. "
            f"{recommendation['name']}"
        )

        print(
            "   Final score:",
            recommendation["score"]
        )

        print(
            "   Content similarity:",
            recommendation[
                "content_similarity"
            ]
        )

        print(
            "   Preference score:",
            recommendation[
                "preference_score"
            ]
        )

        print(
            "   Quality score:",
            recommendation[
                "quality_score"
            ]
        )

        print(
            "   Why recommended:"
        )

        for reason in recommendation[
            "reasons"
        ]:

            print(
                "   ✓",
                reason
            )


    print(
        "\n========================================"
    )

    print(
        "RECOMMENDATION COMPLETE"
    )

    print(
        "========================================"
    )