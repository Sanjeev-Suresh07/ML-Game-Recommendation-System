import pandas as pd

games = pd.read_csv("dataset/games.csv")

print("Original Shape:")
print(games.shape)

# Remove the Movies column
games = games.drop(columns=["Movies"])

print("\nNew Shape:")
print(games.shape)

print("\nRemaining Columns:")
print(games.columns)