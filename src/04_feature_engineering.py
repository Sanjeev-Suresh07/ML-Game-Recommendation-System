import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
games = pd.read_csv("dataset/games.csv")

# Remove unnecessary column
games = games.drop(columns=["Movies"])

# Fill missing values
features = ["About the game", "Genres", "Tags", "Categories"]

for column in features:
    games[column] = games[column].fillna("")

# Combine important game information
games["combined_features"] = (
    games["About the game"] + " " +
    games["Genres"] + " " +
    games["Tags"] + " " +
    games["Categories"]
)

# Convert text into numerical TF-IDF features
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

tfidf_matrix = tfidf.fit_transform(games["combined_features"])

print("Number of games:", len(games))
print("TF-IDF matrix shape:", tfidf_matrix.shape)