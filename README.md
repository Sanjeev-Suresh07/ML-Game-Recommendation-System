# SIDEQUEST — ML Game Recommendation System

A polished Streamlit app for browsing the Steam catalog by genre. SIDEQUEST uses content-based machine learning to rank genre picks, then finds games similar to a title you like.

## What’s included

- A game-inspired, dark discovery page with Steam cover art and descriptions.
- A sign-in screen with guest browsing for local previews.
- Genre browsing, title search, and helpful sorting.
- A TF-IDF model that compares descriptions, genres, categories, and community tags.
- Cosine similarity picks for a selected genre, plus title-to-title recommendations.
- A small preview catalog for trying the interface before adding the full CSV.

## Run locally

1. Install Python 3.10 or newer.
2. In this folder, install the packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. The Steam dataset is already in `dataset/games.csv` in the working project copy. If you downloaded the ZIP, get the cleaned CSV from Kaggle, rename it to `games.csv`, and put it in `dataset/`.
4. Start the app:

   ```bash
   streamlit run app.py
   ```

The catalog loads automatically from `dataset/games.csv`. The first launch reads the file and prepares the recommendation index; later interactions use the cached model.

## Sign-in preview

The sign-in screen is currently a front-end prototype. Any non-empty email and password will enter the app, or choose “Browse as guest.” It does not create accounts, verify credentials, or save passwords. Add a real authentication provider before using it for real user accounts.

## How the model works

The recommender turns each game's genres, categories, community tags, and a short description into a weighted TF-IDF profile, then compares profiles with cosine similarity. For a selected genre, it ranks games against that genre's average profile. Choosing “Find games like this” compares the selected title against the recommendation index. To keep startup responsive, the app indexes a representative set of up to 25,000 popular and randomly sampled games while still browsing the full catalog. This is content-based: it does not learn from a visitor's personal account or play history.

## Project layout

```text
app.py                 Streamlit interface
src/recommender.py     Data preparation and TF-IDF recommendation model
dataset/games.csv      Local Steam game catalog (kept out of Git and ZIP)
```

## Dataset source

The app uses the [Steam Games Dataset 2025 on Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset), a 90,000+ game catalog scraped from Steam and Steam Spy. Kaggle lists the dataset under the MIT license. The cleaned CSV (about 469 MB) is present in this working project copy, but is excluded from Git and the shareable ZIP to keep them manageable. When using a Git checkout or the ZIP, download `games_march2025_cleaned.csv`, rename it to `games.csv`, and put it in `dataset/`.

The built-in preview catalog is only for trying the interface; it is not a substitute for the full dataset.
