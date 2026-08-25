import pandas as pd
import numpy as np
import re


# ============================================================
# 1. LOAD SAVED GAME DATA
# ============================================================

print("Loading saved game data...")

games = pd.read_pickle(
    "models/games_data.pkl"
)

print(
    "Games loaded:",
    len(games)
)


# ============================================================
# 2. CONVERT REVIEW COLUMNS TO NUMERIC
# ============================================================

games["Positive"] = pd.to_numeric(
    games["Positive"],
    errors="coerce"
).fillna(0)

games["Negative"] = pd.to_numeric(
    games["Negative"],
    errors="coerce"
).fillna(0)


# ============================================================
# 3. POSITIVE REVIEW RATIO
# ============================================================

total_reviews = (
    games["Positive"] +
    games["Negative"]
)


games["positive_ratio_v2"] = np.where(
    total_reviews > 0,
    games["Positive"] / total_reviews,
    0
)


# ============================================================
# 4. BAYESIAN REVIEW SCORE
# ============================================================

# Minimum number of reviews needed before
# trusting the positive ratio strongly.

confidence = 100


global_ratio = (
    games["Positive"].sum()
    /
    max(
        games["Positive"].sum()
        +
        games["Negative"].sum(),
        1
    )
)


games["bayesian_review_score"] = (

    (
        total_reviews /
        (total_reviews + confidence)
    )
    *
    games["positive_ratio_v2"]

    +

    (
        confidence /
        (total_reviews + confidence)
    )
    *
    global_ratio

)


# ============================================================
# 5. METACRITIC SCORE
# ============================================================

games["metacritic_v2"] = pd.to_numeric(
    games["Metacritic score"],
    errors="coerce"
)

games["metacritic_v2"] = (
    games["metacritic_v2"]
    .fillna(0)
)


# Convert 0-100 to 0-1

games["metacritic_normalized"] = (
    games["metacritic_v2"] / 100
)


# ============================================================
# 6. POPULARITY FROM ESTIMATED OWNERS
# ============================================================

def parse_owner_range(value):

    if pd.isna(value):

        return 0.0


    text = str(value)


    numbers = re.findall(
        r"\d[\d,]*",
        text
    )


    if len(numbers) == 0:

        return 0.0


    try:

        lower = float(
            numbers[0].replace(
                ",",
                ""
            )
        )


        if len(numbers) > 1:

            upper = float(
                numbers[1].replace(
                    ",",
                    ""
                )
            )

            return (
                lower + upper
            ) / 2


        return lower

    except:

        return 0.0


games["owners_numeric"] = (
    games["Estimated owners"]
    .apply(parse_owner_range)
)


# Log transform because owner counts
# can vary enormously.

games["owners_log"] = np.log1p(
    games["owners_numeric"]
)


# Normalize popularity

max_owners = games[
    "owners_log"
].max()


if max_owners > 0:

    games["popularity_score"] = (
        games["owners_log"]
        /
        max_owners
    )

else:

    games["popularity_score"] = 0


# ============================================================
# 7. PEAK CCU POPULARITY
# ============================================================

games["Peak CCU"] = pd.to_numeric(
    games["Peak CCU"],
    errors="coerce"
).fillna(0)


games["ccu_log"] = np.log1p(
    games["Peak CCU"]
)


max_ccu = games[
    "ccu_log"
].max()


if max_ccu > 0:

    games["ccu_score"] = (
        games["ccu_log"]
        /
        max_ccu
    )

else:

    games["ccu_score"] = 0


# ============================================================
# 8. COMBINED QUALITY SCORE
# ============================================================

games["quality_score_v2"] = (

    0.60 *
    games["bayesian_review_score"]

    +

    0.20 *
    games["metacritic_normalized"]

    +

    0.10 *
    games["popularity_score"]

    +

    0.10 *
    games["ccu_score"]

)


# ============================================================
# 9. REVIEW CONFIDENCE
# ============================================================

games["review_count_v2"] = total_reviews

games["has_reviews_v2"] = (
    total_reviews > 0
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print()
print(
    "========================================"
)

print(
    "V2 FEATURE ENGINEERING"
)

print(
    "========================================"
)


print(
    "\nAverage review ratio:",
    round(
        games[
            "positive_ratio_v2"
        ].mean(),
        4
    )
)


print(
    "Average Bayesian score:",
    round(
        games[
            "bayesian_review_score"
        ].mean(),
        4
    )
)


print(
    "Average quality score:",
    round(
        games[
            "quality_score_v2"
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
        "review_count_v2",
        "bayesian_review_score",
        "metacritic_normalized",
        "popularity_score",
        "quality_score_v2"
    ]
].sort_values(
    "quality_score_v2",
    ascending=False
).head(10)


print(
    top_games.to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE V2 DATA
# ============================================================

games.to_pickle(
    "models/games_data_v2.pkl"
)


print()
print(
    "Saved:"
)

print(
    "models/games_data_v2.pkl"
)


# ============================================================
# 12. COMPLETE
# ============================================================

print()
print(
    "========================================"
)

print(
    "V2 FEATURE ENGINEERING COMPLETE"
)

print(
    "========================================"
)