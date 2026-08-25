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

print("Dataset shape:", games.shape)


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
# 3. CLEAN TEXT COLUMNS
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
# 4. CREATE CONTENT FEATURES
# ============================================================

# Give Genres and Tags extra importance.

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

print(
    "TF-IDF matrix shape:",
    tfidf_matrix.shape
)


# ============================================================
# 6. CLEAN REVIEW DATA
# ============================================================

review_columns = [
    "Positive",
    "Negative"
]

for column in review_columns:

    games[column] = pd.to_numeric(
        games[column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# 7. REVIEW QUALITY SCORE
# ============================================================

total_reviews = (
    games["Positive"] +
    games["Negative"]
)

games["has_reviews"] = (
    total_reviews > 0
)

games["positive_ratio"] = np.where(
    total_reviews > 0,
    games["Positive"] / total_reviews,
    0
)

games["review_strength"] = np.log1p(
    total_reviews
)

max_strength = games["review_strength"].max()

if max_strength > 0:

    games["review_strength"] = (
        games["review_strength"] /
        max_strength
    )

games["quality_score"] = (
    games["positive_ratio"] *
    games["review_strength"]
)


# ============================================================
# 8. USER PROFILE
# ============================================================

user_profile = {

    "name": "User",

    "games_played": [
        "Counter-Strike 2",
        "PUBG: BATTLEGROUNDS"
    ],

    "preferred_genres": [
        "Action",
        "Shooter",
        "Multiplayer"
    ],

    "preferred_tags": [
        "FPS",
        "Competitive",
        "Online Multiplayer"
    ]
}


# ============================================================
# 9. DISPLAY USER PROFILE
# ============================================================

print("\n================ USER PROFILE ================")

print("User:", user_profile["name"])

print("\nGames Played:")

for game in user_profile["games_played"]:
    print("-", game)

print("\nPreferred Genres:")

for genre in user_profile["preferred_genres"]:
    print("-", genre)

print("\nPreferred Tags:")

for tag in user_profile["preferred_tags"]:
    print("-", tag)


# ============================================================
# 10. FIND PLAYED GAMES
# ============================================================

played_indices = []

print("\n================ GAME CHECK ================")

for played_game in user_profile["games_played"]:

    matches = games[
        games["Name"].str.lower() ==
        played_game.lower()
    ]

    if not matches.empty:

        index = matches.index[0]

        played_indices.append(index)

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
    len(played_indices)
)


# ============================================================
# 11. CREATE USER PROFILE VECTOR
# ============================================================

if len(played_indices) == 0:

    print("\nNo played games found.")

else:

    user_vector = tfidf_matrix[
        played_indices
    ].mean(axis=0)

    user_vector = np.asarray(
        user_vector
    )


    # ========================================================
    # 12. CONTENT SIMILARITY
    # ========================================================

    content_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()


    # ========================================================
    # 13. USER PREFERENCE MATCH
    # ========================================================

    preferred_genres = [
        x.lower().strip()
        for x in user_profile["preferred_genres"]
    ]

    preferred_tags = [
        x.lower().strip()
        for x in user_profile["preferred_tags"]
    ]

    genre_scores = []
    tag_scores = []
    preference_scores = []


    for _, row in games.iterrows():

        # Get actual genre and tag text
        genres = str(row["Genres"]).lower()
        tags = str(row["Tags"]).lower()

        # Convert comma-separated values into lists
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


        # ----------------------------------------------------
        # Genre matching
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Tag matching
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Combined preference score
        # ----------------------------------------------------

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


    genre_scores = np.array(
        genre_scores
    )

    tag_scores = np.array(
        tag_scores
    )

    preference_scores = np.array(
        preference_scores
    )


    # ========================================================
    # 14. QUALITY SCORE
    # ========================================================

    quality_scores = games[
        "quality_score"
    ].values.copy()

    has_reviews = games[
        "has_reviews"
    ].values


    # Normalize only games with reviews

    reviewed_quality = quality_scores[
        has_reviews
    ]

    if len(reviewed_quality) > 0:

        max_quality = reviewed_quality.max()

        if max_quality > 0:

            quality_scores[
                has_reviews
            ] = (
                quality_scores[
                    has_reviews
                ] / max_quality
            )


    # ========================================================
    # 15. FINAL SCORE
    # ========================================================

    # Main signal: content similarity
    # Secondary: user preferences
    # Small bonus: review quality

    final_scores = (
        0.70 * content_scores +
        0.25 * preference_scores
    )


    # Add review score ONLY where review data exists

    final_scores += np.where(
        has_reviews,
        0.05 * quality_scores,
        0
    )


    # ========================================================
    # 16. REMOVE ALREADY PLAYED GAMES
    # ========================================================

    final_scores[
        played_indices
    ] = -1


    # ========================================================
    # 17. GET TOP RECOMMENDATIONS
    # ========================================================

    top_indices = np.argsort(
        final_scores
    )[::-1][:5]


    # ========================================================
    # 18. DISPLAY RECOMMENDATIONS
    # ========================================================

    print(
        "\n================ RECOMMENDATIONS ================"
    )


    for number, index in enumerate(
        top_indices,
        start=1
    ):

        print(
            f"\n{number}. "
            f"{games.loc[index, 'Name']}"
        )

        print(
            f"   Content similarity : "
            f"{content_scores[index]:.4f}"
        )

        print(
            f"   Genre match        : "
            f"{genre_scores[index]:.4f}"
        )

        print(
            f"   Tag match          : "
            f"{tag_scores[index]:.4f}"
        )

        print(
            f"   Preference score   : "
            f"{preference_scores[index]:.4f}"
        )


        if has_reviews[index]:

            print(
                f"   Review quality     : "
                f"{quality_scores[index]:.4f}"
            )

        else:

            print(
                "   Review quality     : N/A"
            )


        print(
            f"   FINAL SCORE        : "
            f"{final_scores[index]:.4f}"
        )


        # ----------------------------------------------------
        # Recommendation explanation
        # ----------------------------------------------------

        reasons = []


        if content_scores[index] >= 0.50:

            reasons.append(
                "similar game content"
            )


        if genre_scores[index] > 0:

            reasons.append(
                "preferred genre"
            )


        if tag_scores[index] > 0:

            reasons.append(
                "preferred tag"
            )


        if has_reviews[index]:

            reasons.append(
                "review data available"
            )


        print(
            "   Why recommended:"
        )


        if reasons:

            for reason in reasons:

                print(
                    "   ✓",
                    reason
                )

        else:

            print(
                "   ✓ content similarity"
            )


    # ========================================================
    # 19. RANKING CHECK
    # ========================================================

    print(
        "\n================ RANKING CHECK ================"
    )

    for number, index in enumerate(
        top_indices,
        start=1
    ):

        print(
            f"{number}. "
            f"{games.loc[index, 'Name']} "
            f"-> {final_scores[index]:.4f}"
        )


    # ========================================================
    # 20. PROFILE SUMMARY
    # ========================================================

    print(
        "\n================ PROFILE SUMMARY ================"
    )

    print(
        "User:",
        user_profile["name"]
    )

    print(
        "Games played:",
        len(played_indices)
    )

    print(
        "Preferred genres:",
        ", ".join(
            user_profile["preferred_genres"]
        )
    )

    print(
        "Preferred tags:",
        ", ".join(
            user_profile["preferred_tags"]
        )
    )