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
# 3. TEXT FEATURES
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
    )


# ============================================================
# 4. COMBINE GAME INFORMATION
# ============================================================

games["combined_features"] = (
    games["About the game"] + " " +
    games["Genres"] + " " +
    games["Tags"] + " " +
    games["Categories"]
)

print("\nExample combined features:")
print(
    games[["Name", "combined_features"]]
    .head(3)
    .to_string(index=False)
)


# ============================================================
# 5. TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = tfidf.fit_transform(
    games["combined_features"]
)

print("\nTF-IDF matrix shape:", tfidf_matrix.shape)


# ============================================================
# 6. CLEAN NUMERIC FEATURES
# ============================================================

numeric_columns = [
    "Positive",
    "Negative",
    "Metacritic score",
    "User score",
    "Average playtime forever",
    "Average playtime two weeks"
]

for column in numeric_columns:
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

    if len(matches) > 0:

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
# 11. CREATE PERSONALIZED RECOMMENDATIONS
# ============================================================

if len(played_indices) == 0:

    print(
        "\nNo played games were found in the dataset."
    )

    print(
        "Cannot create personalized recommendations."
    )

else:

    # --------------------------------------------------------
    # USER PROFILE VECTOR
    # --------------------------------------------------------

    user_profile_vector = tfidf_matrix[
        played_indices
    ].mean(axis=0)

    user_profile_vector = np.asarray(
        user_profile_vector
    )


    # --------------------------------------------------------
    # CONTENT SIMILARITY
    # --------------------------------------------------------

    similarities = cosine_similarity(
        user_profile_vector,
        tfidf_matrix
    ).flatten()


    # --------------------------------------------------------
    # USER PREFERENCE MATCH
    # --------------------------------------------------------

    preference_scores = []

    preferred_genres = [
        x.lower()
        for x in user_profile["preferred_genres"]
    ]

    preferred_tags = [
        x.lower()
        for x in user_profile["preferred_tags"]
    ]

    for index, row in games.iterrows():

        text = (
            str(row["Genres"]) + " " +
            str(row["Tags"]) + " " +
            str(row["Categories"])
        ).lower()

        genre_matches = sum(
            genre in text
            for genre in preferred_genres
        )

        tag_matches = sum(
            tag in text
            for tag in preferred_tags
        )

        total_preferences = (
            len(preferred_genres) +
            len(preferred_tags)
        )

        if total_preferences > 0:

            preference_score = (
                genre_matches +
                tag_matches
            ) / total_preferences

        else:

            preference_score = 0

        preference_scores.append(
            preference_score
        )


    preference_scores = np.array(
        preference_scores
    )


    # --------------------------------------------------------
    # QUALITY SCORE
    # --------------------------------------------------------

    quality_scores = games[
        "quality_score"
    ].values

    max_quality = quality_scores.max()

    if max_quality > 0:

        quality_scores = (
            quality_scores /
            max_quality
        )


    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    final_scores = (

        0.70 * similarities +

        0.20 * preference_scores +

        0.10 * quality_scores

    )


    # --------------------------------------------------------
    # REMOVE PLAYED GAMES
    # --------------------------------------------------------

    final_scores[played_indices] = -1


    # --------------------------------------------------------
    # TOP 5 RECOMMENDATIONS
    # --------------------------------------------------------

    top_indices = np.argsort(
        final_scores
    )[::-1][:5]


    # --------------------------------------------------------
    # DISPLAY RECOMMENDATIONS
    # --------------------------------------------------------

    print(
        "\n================ PERSONALIZED "
        "RECOMMENDATIONS ================"
    )

    for number, index in enumerate(
        top_indices,
        start=1
    ):

        print(
            f"{number}. "
            f"{games.loc[index, 'Name']} "
            f"(Score: {final_scores[index]:.4f})"
        )


    # --------------------------------------------------------
    # PROFILE SUMMARY
    # --------------------------------------------------------

    print(
        "\n================ PROFILE SUMMARY ================"
    )

    print(
        "User:",
        user_profile["name"]
    )

    print(
        "Played games:",
        len(user_profile["games_played"])
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