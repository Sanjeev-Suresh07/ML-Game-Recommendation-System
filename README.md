# ML-Based Personalized Game Recommendation System

This team project explores recommendation models for Steam games and includes **SIDEQUEST**, a game-inspired web app for discovering titles by genre and finding games similar to a favorite.

## What you can do

- Browse and search the Steam catalog with game artwork, a cursor spotlight reveal, and ambient cyber-green effects.
- Choose a genre to get content-based recommendations, or request titles similar to a selected game.
- Explore the project's data preparation, feature engineering, KNN, hybrid recommendation, model evaluation, and API experiments in `src/`.

## Run the web app

1. Install Python 3.10 or newer.
2. Install the web app requirements:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start SIDEQUEST:

   ```bash
   streamlit run app.py
   ```

The repository includes a compressed catalog of 89,618 cleaned Steam games at `dataset/games.csv.gz`. For local use, the app also reads `dataset/games.csv` if you have the original CSV instead. If neither file is present, it uses a small built-in preview catalog.

The broader course-project scripts use the extra packages listed in `requirements-project.txt`.

### Dataset credit

The Steam catalog is from [Steam Games Dataset 2025 by Artemiy Ermilov and collaborators](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset), cleaned version as of March 2025. The Kaggle page lists the dataset under the MIT License.

### Sign-in preview

The sign-in screen is a front-end prototype. Any non-empty email and password will continue to the app, or choose “Browse as guest.” It does not create accounts, check passwords, or save credentials. Connect an authentication provider before using real accounts.

## Recommendation model

SIDEQUEST builds weighted TF-IDF profiles from game descriptions, genres, categories, and community tags, then compares them with cosine similarity. Genre recommendations match games to a genre profile; title recommendations match the selected game. The web app indexes a representative sample of up to 8,000 games for responsive recommendations while browsing the full catalog. This content-based model does not use a visitor's account or play history.

The numbered scripts in `src/` document the broader course project, including data exploration and cleaning, engineered features, KNN and hybrid approaches, evaluation, and API work.

## Project structure

```text
app.py                 SIDEQUEST Streamlit interface
src/recommender.py     Content-based model used by the web app
src/01_...19_*.py      Project data, modeling, evaluation, and API scripts
requirements-app.txt      Streamlit and Pandas dependencies for SIDEQUEST
requirements-project.txt  Extra packages for the course-project scripts
dataset/games.csv.gz      Compressed Steam catalog used by the local and cloud app
```

## Team

| Initials | Name |
| --- | --- |
| S | Sanjeev Suresh |
| MS | Jai Harish |

B.Tech – Artificial Intelligence & Data Science
