from importlib.machinery import SourceFileLoader

# Load the final recommender
recommender = SourceFileLoader(
    "recommender",
    "src/11_final_recommender.py"
).load_module()

recommend_games = recommender.recommend_games


# ============================================================
# PROFILE 1 - FPS / COMPETITIVE
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
# PROFILE 2 - RPG / ADVENTURE
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
# PROFILE 3 - STRATEGY / CASUAL
# ============================================================

profile_3_games = [
    "Civilization VI"
]

profile_3_genres = [
    "Strategy",
    "Simulation"
]

profile_3_tags = [
    "Turn-Based",
    "Strategy",
    "Singleplayer"
]


# ============================================================
# TEST FUNCTION
# ============================================================

def test_profile(
    profile_name,
    played_games,
    genres,
    tags
):

    print("\n========================================")
    print(profile_name)
    print("========================================")

    print("\nPlayed games:")

    for game in played_games:
        print("-", game)

    recommendations = recommend_games(
        played_games,
        genres,
        tags,
        top_k=5
    )

    print("\nRecommendations:")

    for number, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{number}. "
            f"{recommendation['name']} "
            f"-> "
            f"{recommendation['score']}"
        )


# ============================================================
# RUN TESTS
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
    "PROFILE 3 - STRATEGY / CASUAL",
    profile_3_games,
    profile_3_genres,
    profile_3_tags
)


print(
    "\n========================================"
)

print(
    "MULTI-PROFILE TEST COMPLETE"
)

print(
    "========================================"
)