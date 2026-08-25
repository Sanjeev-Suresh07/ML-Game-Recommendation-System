import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD V2 DATA
# ============================================================

print("Loading V2 game data...")

games = pd.read_pickle(
    "models/games_data_v2.pkl"
)

print(
    "Games loaded:",
    len(games)
)


# ============================================================
# 2. REVIEW DATA
# ============================================================

games["Positive"] = pd.to_numeric(
    games["Positive"],
    errors="coerce"
).fillna(0)

games["Negative"] = pd.to_numeric(
    games["Negative"],
    errors="coerce"
).fillna(0)


games["review_count_v3"] = (
    games["Positive"] +
    games["Negative"]
)


# ============================================================
# 3. POSITIVE REVIEW RATIO
# ============================================================

games["positive_ratio_v3"] = np.where(

    games["review_count_v3"] > 0,

    games["Positive"] /
    games["review_count_v3"],

    0

)


# ============================================================
# 4. GLOBAL REVIEW RATIO
# ============================================================

total_positive = games["Positive"].sum()

total_negative = games["Negative"].sum()

total_all = (
    total_positive +
    total_negative
)


if total_all > 0:

    global_ratio = (
        total_positive /
        total_all
    )

else:

    global_ratio = 0.5


print(
    "\nGlobal positive ratio:",
    round(global_ratio, 4)
)


# ============================================================
# 5. BAYESIAN REVIEW SCORE
# ============================================================

# Higher m means we require more reviews
# before completely trusting the game's ratio.

m = 1000


games["bayesian_score_v3"] = (

    (
        games["review_count_v3"] /
        (
            games["review_count_v3"] + m
        )
    )
    *
    games["positive_ratio_v3"]

    +

    (
        m /
        (
            games["review_count_v3"] + m
        )
    )
    *
    global_ratio

)


# ============================================================
# 6. REVIEW CONFIDENCE
# ============================================================

# Logarithmic growth prevents extremely popular
# games from completely dominating.

games["review_confidence_v3"] = (

    np.log1p(
        games["review_count_v3"]
    )

    /

    np.log1p(
        games["review_count_v3"].max()
    )

)


# ============================================================
# 7. METACRITIC
# ============================================================

games["metacritic_v3"] = pd.to_numeric(
    games["Metacritic score"],
    errors="coerce"
).fillna(0)


games["metacritic_score_v3"] = (

    games["metacritic_v3"] / 100

)


# ============================================================
# 8. POPULARITY
# ============================================================

# games_data_v2 already contains popularity_score

if "popularity_score" in games.columns:

    games["popularity_v3"] = (
        games["popularity_score"]
        .fillna(0)
    )

else:

    games["popularity_v3"] = 0


# ============================================================
# 9. PEAK CCU
# ============================================================

if "ccu_score" in games.columns:

    games["ccu_v3"] = (
        games["ccu_score"]
        .fillna(0)
    )

else:

    games["ccu_v3"] = 0


# ============================================================
# 10. FINAL QUALITY SCORE
# ============================================================

games["quality_score_v3"] = (

    0.45 *
    games["bayesian_score_v3"]

    +

    0.20 *
    games["review_confidence_v3"]

    +

    0.20 *
    games["metacritic_score_v3"]

    +

    0.10 *
    games["popularity_v3"]

    +

    0.05 *
    games["ccu_v3"]

)


# ============================================================
# 11. KEEP SCORE BETWEEN 0 AND 1
# ============================================================

games["quality_score_v3"] = (
    games["quality_score_v3"]
    .clip(0, 1)
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print()
print(
    "========================================"
)

print(
    "IMPROVED QUALITY SCORE"
)

print(
    "========================================"
)


print(
    "\nAverage quality score:",
    round(
        games[
            "quality_score_v3"
        ].mean(),
        4
    )
)


print(
    "\nTop quality games:"
)


top_games = games[
    [
        "Name",
        "Positive",
        "Negative",
        "review_count_v3",
        "positive_ratio_v3",
        "bayesian_score_v3",
        "review_confidence_v3",
        "metacritic_score_v3",
        "popularity_v3",
        "quality_score_v3"
    ]
].sort_values(
    "quality_score_v3",
    ascending=False
).head(15)


print(
    top_games.to_string(
        index=False
    )
)


# ============================================================
# 13. CHECK REVIEW COUNTS
# ============================================================

print()
print(
    "Review count statistics:"
)

print(
    games[
        "review_count_v3"
    ].describe()
)


# ============================================================
# 14. SAVE V3 DATA
# ============================================================

games.to_pickle(
    "models/games_data_v3.pkl"
)


print()
print(
    "Saved:"
)

print(
    "models/games_data_v3.pkl"
)


# ============================================================
# 15. COMPLETE
# ============================================================

print()
print(
    "========================================"
)

print(
    "QUALITY SCORE V3 COMPLETE"
)

print(
    "========================================"
)