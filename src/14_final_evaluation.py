import pandas as pd
import joblib
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

print("Loading saved recommendation model...")

tfidf_vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

tfidf_matrix = joblib.load(
    "models/tfidf_matrix.pkl"
)

games = pd.read_pickle(
    "models/games_data.pkl"
)

print("Model loaded successfully.")
print("Games available:", len(games))
print("TF-IDF matrix shape:", tfidf_matrix.shape)


# ============================================================
# 2. RECOMMENDATION FUNCTION
# ============================================================

def recommend_games(
    played_games,
    preferred_genres,
    preferred_tags,
    top_k=5
):

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


    if len(matched_indices) == 0:

        print(
            "No played games were found."
        )

        return []


    # Convert dataframe indexes to matrix positions

    matrix_indices = []

    for index in matched_indices:

        matrix_indices.append(
            games.index.get_loc(index)
        )


    # --------------------------------------------------------
    # Create user profile vector
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
    # Preference scores
    # --------------------------------------------------------

    scores = []


    for i in range(len(games)):

        game_name = str(
            games.iloc[i]["Name"]
        ).strip()


        # Skip already played games

        played_lower = [

            x.lower().strip()

            for x in played_games

        ]

        if game_name.lower() in played_lower:

            continue


        genres = str(
            games.iloc[i].get(
                "Genres",
                ""
            )
        ).lower()


        tags = str(
            games.iloc[i].get(
                "Tags",
                ""
            )
        ).lower()


        # ----------------------------------------------------
        # Genre matching
        # ----------------------------------------------------

        genre_matches = 0


        for genre in preferred_genres:

            if genre.lower() in genres:

                genre_matches += 1


        if len(preferred_genres) > 0:

            genre_score = (
                genre_matches /
                len(preferred_genres)
            )

        else:

            genre_score = 0


        # ----------------------------------------------------
        # Tag matching
        # ----------------------------------------------------

        tag_matches = 0


        for tag in preferred_tags:

            if tag.lower() in tags:

                tag_matches += 1


        if len(preferred_tags) > 0:

            tag_score = (
                tag_matches /
                len(preferred_tags)
            )

        else:

            tag_score = 0


        # ----------------------------------------------------
        # Preference score
        # ----------------------------------------------------

        preference_score = (
            genre_score +
            tag_score
        ) / 2


        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        final_score = (

            0.70 *
            similarities[i]

            +

            0.30 *
            preference_score

        )


        scores.append({

            "index": i,

            "score": final_score,

            "similarity": similarities[i],

            "preference": preference_score

        })


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    return scores[:top_k]


# ============================================================
# 3. PRECISION@5
# ============================================================

def calculate_precision_at_5(
    recommendations,
    preferred_genres,
    preferred_tags
):

    if len(recommendations) == 0:

        return 0.0


    relevant = 0


    for recommendation in recommendations:

        index = recommendation["index"]


        genres = str(
            games.iloc[index].get(
                "Genres",
                ""
            )
        ).lower()


        tags = str(
            games.iloc[index].get(
                "Tags",
                ""
            )
        ).lower()


        matched = False


        # Genre check

        for genre in preferred_genres:

            if genre.lower() in genres:

                matched = True
                break


        # Tag check

        if not matched:

            for tag in preferred_tags:

                if tag.lower() in tags:

                    matched = True
                    break


        if matched:

            relevant += 1


    return relevant / 5


# ============================================================
# 4. TEST PROFILES
# ============================================================

profiles = [

    {
        "name":
        "FPS / COMPETITIVE",

        "played_games": [

            "Counter-Strike 2",
            "PUBG: BATTLEGROUNDS"

        ],

        "genres": [

            "Action",
            "Shooter",
            "Multiplayer"

        ],

        "tags": [

            "FPS",
            "Competitive",
            "Online Multiplayer"

        ]
    },


    {
        "name":
        "RPG / ADVENTURE",

        "played_games": [

            "The Witcher 3: Wild Hunt"

        ],

        "genres": [

            "RPG",
            "Adventure"

        ],

        "tags": [

            "Story Rich",
            "Fantasy",
            "Open World"

        ]
    },


    {
        "name":
        "STRATEGY / SIMULATION",

        "played_games": [

            "Countryballs at War"

        ],

        "genres": [

            "Strategy",
            "Simulation"

        ],

        "tags": [

            "Strategy",
            "War",
            "Management"

        ]
    }

]


# ============================================================
# 5. FINAL EVALUATION
# ============================================================

print(
    "\n========================================"
)

print(
    "FINAL MODEL EVALUATION"
)

print(
    "========================================"
)


precision_scores = []


for number, profile in enumerate(
    profiles,
    start=1
):

    print(
        f"\nPROFILE {number} - "
        f"{profile['name']}"
    )

    print(
        "----------------------------------------"
    )


    print(
        "Played games:"
    )


    for game in profile[
        "played_games"
    ]:

        print(
            "-",
            game
        )


    recommendations = recommend_games(

        profile[
            "played_games"
        ],

        profile[
            "genres"
        ],

        profile[
            "tags"
        ],

        top_k=5

    )


    print(
        "\nTop recommendations:"
    )


    if len(recommendations) == 0:

        print(
            "No recommendations generated."
        )

        continue


    for rank, recommendation in enumerate(

        recommendations,

        start=1

    ):

        name = games.iloc[
            recommendation["index"]
        ]["Name"]


        print(

            f"{rank}. {name} "
            f"-> "
            f"{recommendation['score']:.4f}"

        )


    precision = calculate_precision_at_5(

        recommendations,

        profile[
            "genres"
        ],

        profile[
            "tags"
        ]

    )


    precision_scores.append(
        precision
    )


    print(
        f"\nPrecision@5: "
        f"{precision:.2f}"
    )


# ============================================================
# 6. FINAL RESULTS
# ============================================================

print(
    "\n========================================"
)

print(
    "FINAL RESULTS"
)

print(
    "========================================"
)


if len(precision_scores) > 0:

    average_precision = np.mean(
        precision_scores
    )


    print(
        "Profiles tested:",
        len(precision_scores)
    )


    print(
        "Recommendations per profile: 5"
    )


    print(
        f"Average Precision@5: "
        f"{average_precision:.2f}"
    )


    print(
        "\nEvaluation completed successfully."
    )


else:

    print(
        "No profiles could be evaluated."
    )


print(
    "\n========================================"
)

print(
    "FINAL EVALUATION COMPLETE"
)

print(
    "========================================"
)