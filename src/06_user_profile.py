import pandas as pd


# ============================================================
# 1. LOAD DATASET
# ============================================================

games = pd.read_csv("dataset/games.csv", index_col=False)

print("Dataset shape:", games.shape)


# ============================================================
# 2. BASIC CLEANING
# ============================================================

games["Name"] = (
    games["Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Clean text columns
features = [
    "Genres",
    "Tags",
    "Categories"
]

for column in features:

    games[column] = (
        games[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# 3. USER PROFILE
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
# 4. DISPLAY USER PROFILE
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
# 5. CHECK WHICH GAMES EXIST IN DATASET
# ============================================================

print("\n================ GAME CHECK ================")

for game_name in user_profile["games_played"]:

    matches = games[
        games["Name"].str.lower() == game_name.lower()
    ]

    if matches.empty:
        print("Not found:", game_name)

    else:
        print("Found:", matches.iloc[0]["Name"])


# ============================================================
# 6. FIND USER'S PLAYED GAMES
# ============================================================

played_games = games[
    games["Name"].str.lower().isin(
        [
            game.lower()
            for game in user_profile["games_played"]
        ]
    )
]

print("\n================ PLAYED GAMES ================")

print(
    played_games[
        ["AppID", "Name", "Genres", "Tags"]
    ].head(10)
)


# ============================================================
# 7. USER PROFILE SUMMARY
# ============================================================

print("\n================ PROFILE SUMMARY ================")

print(
    "Number of games played:",
    len(played_games)
)

print(
    "Preferred genres:",
    ", ".join(user_profile["preferred_genres"])
)

print(
    "Preferred tags:",
    ", ".join(user_profile["preferred_tags"])
)