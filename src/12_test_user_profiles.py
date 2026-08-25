from importlib.machinery import SourceFileLoader


# ============================================================
# 1. LOAD FINAL RECOMMENDER
# ============================================================

recommender = SourceFileLoader(
    "recommender",
    "src/11_final_recommender.py"
).load_module()

recommend_games = recommender.recommend_games


# ============================================================
# 2. PROFILE 1 - FPS / COMPETITIVE
# ============================================================

profile_1_games = [
    "Counter-Strike 2",
    "PUBG: BATTLEGROUNDS"
]

profile_1_genres = [
    "Action",
    "Shooter",
    "Multiplayer"
]

profile_1_tags = [
    "FPS",
    "Competitive",
    "Online Multiplayer"
]


# ============================================================
# 3. PROFILE 2 - RPG / ADVENTURE
# ============================================================

profile_2_games = [
    "The Witcher 3: Wild Hunt"
]

profile_2_genres = [
    "RPG",
    "Adventure"
]

profile_2_tags = [
    "Story Rich",
    "Fantasy",
    "Open World"
]


# ============================================================
# 4. PROFILE 3 - STRATEGY / SIMULATION
# ============================================================

profile_3_games = [
    "Countryballs at War"
]

profile_3_genres = [
    "Strategy",
    "Simulation"
]

profile_3_tags = [
    "Strategy",
    "Singleplayer"
]


# ============================================================
# 5. TEST PROFILE FUNCTION
# ============================================================

def test_profile(
    profile_name,
    played_games,
    genres,
    tags
):

    print(
        "\n========================================"
    )

    print(
        profile_name
    )

    print(
        "========================================"
    )

    print(
        "\nPlayed games:"
    )

    for game in played_games:
        print("-", game)

    recommendations = recommend_games(
        played_games,
        genres,
        tags,
        top_k=5
    )

    print(
        "\nRecommendations:"
    )

    if len(recommendations) == 0:

        print(
            "No recommendations generated."
        )

        return

    for number, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{number}. "
            f"{recommendation['name']} "
            f"-> {recommendation['score']}"
        )


# ============================================================
# 6. RUN PROFILE TESTS
# ============================================================

test_profile(
    "PROFILE 1 - FPS / COMPETITIVE",
    profile_1_games,
    profile_1_genres,
    profile_1_tags
)

test_profile(
    "PROFILE 2 - RPG / ADVENTURE",
    profile_2_games,
    profile_2_genres,
    profile_2_tags
)

test_profile(
    "PROFILE 3 - STRATEGY / SIMULATION",
    profile_3_games,
    profile_3_genres,
    profile_3_tags
)


# ============================================================
# 7. COMPLETE
# ============================================================

print(
    "\n========================================"
)

print(
    "MULTI-PROFILE TEST COMPLETE"
)

print(
    "========================================"
)