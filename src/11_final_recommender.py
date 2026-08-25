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
# 2. PREFERENCE MAPPING
# ============================================================

preference_mapping = {

    "action": [
        "action"
    ],

    "shooter": [
        "shooter",
        "fps",
        "first-person"
    ],

    "multiplayer": [
        "multi-player",
        "multiplayer",
        "online multiplayer",
        "online pvp",
        "pvp",
        "cross-platform multiplayer",
        "co-op",
        "online co-op"
    ],

    "fps": [
        "fps",
        "shooter",
        "first-person"
    ],

    "competitive": [
        "competitive",
        "pvp",
        "online pvp"
    ],

    "online multiplayer": [
        "online multiplayer",
        "online pvp",
        "multi-player",
        "multiplayer",
        "cross-platform multiplayer"
    ]
}


# ============================================================
# 3. RECOMMENDATION FUNCTION
# ============================================================

def recommend_games(
    played_games,
    preferred_genres=None,
    preferred_tags=None,
    top_k=5
):

    if preferred_genres is None:
        preferred_genres = []

    if preferred_tags is None:
        preferred_tags = []


    # --------------------------------------------------------
    # Find played games
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
    # Create user profile
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
    # Normalize preferences
    # --------------------------------------------------------

    preferred_genres = [
        x.lower().strip()
        for x in preferred_genres
    ]

    preferred_tags = [
        x.lower().strip()
        for x in preferred_tags
    ]


    all_preferences = (
        preferred_genres +
        preferred_tags
    )


    # --------------------------------------------------------
    # Preference scoring
    # --------------------------------------------------------

    preference_scores = []

    genre_scores = []

    tag_scores = []


    for _, row in games.iterrows():

        genres = str(
            row["Genres"]
        ).lower()

        tags = str(
            row["Tags"]
        ).lower()

        categories = str(
            row["Categories"]
        ).lower()


        # Combine metadata
        metadata = (
            genres + " " +
            tags + " " +
            categories
        )


        # -----------------------------------------------
        # Genre score
        # -----------------------------------------------

        genre_matches = 0

        for preference in preferred_genres:

            possible_matches = (
                preference_mapping.get(
                    preference,
                    [preference]
                )
            )

            if any(
                value in metadata
                for value in possible_matches
            ):

                genre_matches += 1


        if len(preferred_genres) > 0:

            genre_score = (
                genre_matches /
                len(preferred_genres)
            )

        else:

            genre_score = 0


        # -----------------------------------------------
        # Tag score
        # -----------------------------------------------

        tag_matches = 0

        for preference in preferred_tags:

            possible_matches = (
                preference_mapping.get(
                    preference,
                    [preference]
                )
            )

            if any(
                value in metadata
                for value in possible_matches
            ):

                tag_matches += 1


        if len(preferred_tags) > 0:

            tag_score = (
                tag_matches /
                len(preferred_tags)
            )

        else:

            tag_score = 0


        # -----------------------------------------------
        # Combined preference score
        # -----------------------------------------------

        preference_score = (
            0.6 * genre_score +
            0.4 * tag_score
        )


        genre_scores.append(
            genre_score
        )

        tag_scores.append(
            tag_score
        )

        preference_scores.append(
            preference_score
        )


    preference_scores = np.array(
        preference_scores
    )

    genre_scores = np.array(
        genre_scores
    )

    tag_scores = np.array(
        tag_scores
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

        0.65 * content_scores +

        0.30 * preference_scores +

        0.05 * quality_scores

    )


    # --------------------------------------------------------
    # Remove already played games
    # --------------------------------------------------------

    final_scores[
        played_indices
    ] = -1


    # --------------------------------------------------------
    # Candidate games
    # --------------------------------------------------------

    candidate_count = min(
        top_k * 5,
        len(games)
    )

    candidate_indices = np.argsort(
        final_scores
    )[::-1][:candidate_count]


    # --------------------------------------------------------
    # Remove duplicate names
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
        # Explanation
        # ----------------------------------------------------

        reasons = []


        if content_scores[index] >= 0.50:

            reasons.append(
                "similar to your played games"
            )


        if genre_scores[index] > 0:

            reasons.append(
                "matches your preferred genres"
            )


        if tag_scores[index] > 0:

            reasons.append(
                "matches your preferred play style"
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

            "genre_score": round(
                float(genre_scores[index]),
                4
            ),

            "tag_score": round(
                float(tag_scores[index]),
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
# 4. TEST FINAL RECOMMENDER
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


    recommendations = recommend_games(

        played_games,

        preferred_genres,

        preferred_tags,

        top_k=5

    )


    # --------------------------------------------------------
    # Display
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
            "   Genre score:",
            recommendation[
                "genre_score"
            ]
        )

        print(
            "   Tag score:",
            recommendation[
                "tag_score"
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