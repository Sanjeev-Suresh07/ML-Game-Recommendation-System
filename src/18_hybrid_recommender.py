import sys

# Allow importing File 15
sys.path.append("src")

from importlib import import_module

recommendation_api = import_module(
    "15_recommendation_api"
)

recommend_games = recommendation_api.recommend_games


# ============================================================
# EVALUATION PROFILES
# ============================================================

profiles = [

    {
        "name": "FPS / COMPETITIVE",

        "played_games": [
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
    },


    {
        "name": "RPG / ADVENTURE",

        "played_games": [
            "The Witcher 3: Wild Hunt"
        ],

        "preferred_genres": [
            "RPG",
            "Adventure",
            "Action"
        ],

        "preferred_tags": [
            "RPG",
            "Story Rich",
            "Adventure"
        ]
    },


    {
        "name": "STRATEGY / SIMULATION",

        "played_games": [
            "Countryballs at War"
        ],

        "preferred_genres": [
            "Strategy",
            "Simulation",
            "Indie"
        ],

        "preferred_tags": [
            "Strategy",
            "Simulation",
            "Singleplayer"
        ]
    }

]


# ============================================================
# PRECISION FUNCTION
# ============================================================

def calculate_precision(
    recommendations,
    preferred_genres
):

    if len(recommendations) == 0:

        return 0.0


    relevant = 0


    for game in recommendations:

        name = game["name"].lower()

        # Use recommendation reasons to determine
        # whether it matches the user's preferences.

        reasons = game["reason"]

        matches_preference = False

        for reason in reasons:

            if (
                "genre" in reason.lower()
                or
                "play style" in reason.lower()
            ):

                matches_preference = True


        if matches_preference:

            relevant += 1


    return relevant / len(
        recommendations
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

print()
print("========================================")
print("HYBRID MODEL EVALUATION")
print("========================================")


all_precisions = []


for profile_number, profile in enumerate(
    profiles,
    start=1
):

    print()
    print(
        f"PROFILE {profile_number} - "
        f"{profile['name']}"
    )

    print("----------------------------------------")


    print("Played games:")

    for game in profile[
        "played_games"
    ]:

        print(
            "-",
            game
        )


    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    response = recommend_games(

        profile["played_games"],

        profile["preferred_genres"],

        profile["preferred_tags"],

        top_k=5

    )


    if not response["success"]:

        print(
            "\nRecommendation failed:"
        )

        print(
            response["message"]
        )

        continue


    recommendations = response[
        "recommendations"
    ]


    # --------------------------------------------------------
    # Display recommendations
    # --------------------------------------------------------

    print(
        "\nTop recommendations:"
    )


    for number, game in enumerate(

        recommendations,

        start=1

    ):

        print(
            f"{number}. "
            f"{game['name']} "
            f"-> "
            f"{game['score']:.4f}"
        )


    # --------------------------------------------------------
    # Calculate Precision@5
    # --------------------------------------------------------

    precision = calculate_precision(

        recommendations,

        profile[
            "preferred_genres"
        ]

    )


    all_precisions.append(
        precision
    )


    print()

    print(
        f"Precision@5: "
        f"{precision:.2f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("========================================")
print("FINAL HYBRID RESULTS")
print("========================================")


if len(all_precisions) > 0:

    average_precision = (

        sum(all_precisions)

        /

        len(all_precisions)

    )


else:

    average_precision = 0.0


print(
    "Profiles tested:",
    len(all_precisions)
)


print(
    "Recommendations per profile: 5"
)


print(
    f"Average Precision@5: "
    f"{average_precision:.2f}"
)


print()
print("========================================")
print("HYBRID MODEL EVALUATION COMPLETE")
print("========================================")