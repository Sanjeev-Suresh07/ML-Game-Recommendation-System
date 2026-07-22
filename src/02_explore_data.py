import pandas as pd

games = pd.read_csv("dataset/games.csv")

print("Dataset Shape:")
print(games.shape)

print("\nColumn Names:")
print(games.columns)

print("\nDataset Information:")
print(games.info())