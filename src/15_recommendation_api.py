import pandas as pd
import joblib
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

print("Loading recommendation model...")

tfidf_vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

tfidf_matrix = joblib.load(
    "models/tfidf_matrix.pkl"
)

# Use V3 data for improved quality scores
games = pd.read_pickle(
    "models/games_data_v3.pkl"
)

print("Recommendation model loaded.")
print("Games available:", len(games))


# ============================================================
# 2. PREPARE QUALITY AND POPULARITY
# ============================================================

# Make sure required columns exist

if "quality_score_v3" not in games.columns:

    games["quality_score_v3"] = 0.0


if "popularity_v3" not in games.columns:

    games["popularity_v3"] = 0.0


games["quality_score_v3"] = pd.to_numeric(
    games["quality_score_v3"],
    errors="coerce"
).fillna(0)


games["popularity_v3"] = pd.to_numeric(
    games["popularity_v3"],
    errors="coerce"
).fillna(0)


# ============================================================
# 3. RECOMMENDATION FUNCTION
# ============================================================

def recommend_games(
    played_games,
    preferred_genres,
    preferred_tags,
    top_k=5
):

    # --------------------------------------------------------
    # Find played games
    # --------------------------------------------------------

    matched_indices = []

    for game_name in played_games:

        matches = games[
            games["Name"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            game_name.strip().lower()
        ]

        if not matches.empty:

            matched_indices.append(
                matches.index[0]
            )


    # --------------------------------------------------------
    # No games found
    # --------------------------------------------------------

    if len(matched_indices) == 0:

        return {
            "success": False,
            "message": "None of the played games were found.",
            "recommendations": []
        }


    # --------------------------------------------------------
    # Convert dataframe indexes to matrix indexes
    # --------------------------------------------------------

    matrix_indices = []

    for index in matched_indices:

        matrix_indices.append(
            games.index.get_loc(index)
        )


    # --------------------------------------------------------
    # Create user profile
    # --------------------------------------------------------

    user_vector = np.asarray(
        tfidf_matrix[
            matrix_indices
        ].mean(axis=0)
    )


    # --------------------------------------------------------
    # Content similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()


    # --------------------------------------------------------
    # Played game set
    # --------------------------------------------------------

    played_set = set(
        game.strip().lower()
        for game in played_games
    )


    results = []


    # ========================================================
    # SCORE EVERY GAME
    # ========================================================

    for i in range(len(games)):

        game_name = str(
            games.iloc[i]["Name"]
        ).strip()


        # Skip already played games

        if game_name.lower() in played_set:

            continue


        # ----------------------------------------------------
        # Game information
        # ----------------------------------------------------

        genres = str(
            games.iloc[i].get(
                "Genres",
                ""
            )
        )

        tags = str(
            games.iloc[i].get(
                "Tags",
                ""
            )
        )


        genres_lower = genres.lower()
        tags_lower = tags.lower()


        # ----------------------------------------------------
        # Genre score
        # ----------------------------------------------------

        genre_matches = 0

        for genre in preferred_genres:

            if genre.lower() in genres_lower:

                genre_matches += 1


        if len(preferred_genres) > 0:

            genre_score = (
                genre_matches /
                len(preferred_genres)
            )

        else:

            genre_score = 0.0


        # ----------------------------------------------------
        # Tag score
        # ----------------------------------------------------

        tag_matches = 0

        for tag in preferred_tags:

            if tag.lower() in tags_lower:

                tag_matches += 1


        if len(preferred_tags) > 0:

            tag_score = (
                tag_matches /
                len(preferred_tags)
            )

        else:

            tag_score = 0.0


        # ----------------------------------------------------
        # Preference score
        # ----------------------------------------------------

        preference_score = (
            genre_score +
            tag_score
        ) / 2


        # ----------------------------------------------------
        # Content similarity
        # ----------------------------------------------------

        content_score = float(
            similarities[i]
        )


        # ----------------------------------------------------
        # Quality score
        # ----------------------------------------------------

        quality_score = float(
            games.iloc[i][
                "quality_score_v3"
            ]
        )


        # ----------------------------------------------------
        # Popularity score
        # ----------------------------------------------------

        popularity_score = float(
            games.iloc[i][
                "popularity_v3"
            ]
        )


        # ====================================================
        # HYBRID FINAL SCORE
        # ====================================================

        final_score = (

            0.55 * content_score

            +

            0.20 * preference_score

            +

            0.15 * quality_score

            +

            0.10 * popularity_score

        )


        # ----------------------------------------------------
        # Recommendation reasons
        # ----------------------------------------------------

        reasons = []


        if content_score >= 0.50:

            reasons.append(
                "Similar to your played games"
            )


        if genre_score > 0:

            reasons.append(
                "Matches your preferred genres"
            )


        if tag_score > 0:

            reasons.append(
                "Matches your preferred play style"
            )


        if quality_score >= 0.70:

            reasons.append(
                "Highly rated by players"
            )


        if popularity_score >= 0.70:

            reasons.append(
                "Popular among players"
            )


        if len(reasons) == 0:

            reasons.append(
                "Recommended based on your profile"
            )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "name": game_name,

            "score": round(
                final_score,
                4
            ),

            "content_similarity": round(
                content_score,
                4
            ),

            "preference_score": round(
                preference_score,
                4
            ),

            "quality_score": round(
                quality_score,
                4
            ),

            "popularity_score": round(
                popularity_score,
                4
            ),

            "reason": reasons

        })


    # ========================================================
    # SORT RESULTS
    # ========================================================

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    results = results[:top_k]


    # ========================================================
    # RETURN API RESPONSE
    # ========================================================

    return {

        "success": True,

        "message":
        "Hybrid recommendations generated successfully.",

        "played_games":
        played_games,

        "preferred_genres":
        preferred_genres,

        "preferred_tags":
        preferred_tags,

        "recommendations":
        results

    }


# ============================================================
# 4. LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("HYBRID RECOMMENDATION API TEST")
    print("========================================")


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


    response = recommend_games(

        played_games,

        preferred_genres,

        preferred_tags,

        top_k=5

    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print()

    print(
        "Success:",
        response["success"]
    )


    print(
        "\nRecommendations:"
    )


    for number, game in enumerate(

        response["recommendations"],

        start=1

    ):

        print()

        print(
            f"{number}. {game['name']}"
        )

        print(
            "Final score:",
            game["score"]
        )

        print(
            "Content similarity:",
            game["content_similarity"]
        )

        print(
            "Preference score:",
            game["preference_score"]
        )

        print(
            "Quality score:",
            game["quality_score"]
        )

        print(
            "Popularity score:",
            game["popularity_score"]
        )

        print(
            "Why recommended:"
        )


        for reason in game["reason"]:

            print(
                "✓",
                reason
            )


    print()

    print(
        "========================================"
    )

    print(
        "HYBRID RECOMMENDATION API TEST COMPLETE"
    )

    print(
        "========================================"
    )