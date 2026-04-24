from ast import literal_eval
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
CREDITS_PATH = BASE_DIR / "tmdb_5000_credits.csv" / "tmdb_5000_credits.csv"
MOVIES_PATH = BASE_DIR / "tmdb_5000_movies.csv" / "tmdb_5000_movies.csv"


def _convert(obj):
    return [item["name"] for item in literal_eval(obj)]


def _convert_cast(obj):
    names = []
    for item in literal_eval(obj)[:3]:
        names.append(item["name"])
    return names


def _fetch_director(obj):
    for item in literal_eval(obj):
        if item["job"] == "Director":
            return [item["name"]]
    return []


def _load_data():
    credits = pd.read_csv(CREDITS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    merged = movies.merge(credits, on="title")
    merged = merged[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    merged = merged.dropna().copy()

    merged["genres"] = merged["genres"].apply(_convert)
    merged["keywords"] = merged["keywords"].apply(_convert)
    merged["cast"] = merged["cast"].apply(_convert_cast)
    merged["crew"] = merged["crew"].apply(_fetch_director)
    merged["overview"] = merged["overview"].apply(lambda text: text.split())

    for column in ["genres", "keywords", "cast", "crew"]:
        merged[column] = merged[column].apply(
            lambda values: [value.replace(" ", "") for value in values]
        )

    merged["tags"] = (
        merged["overview"]
        + merged["genres"]
        + merged["keywords"]
        + merged["cast"]
        + merged["crew"]
    )

    final_df = merged[["movie_id", "title", "genres", "tags"]].copy()
    final_df["genres"] = final_df["genres"].apply(lambda values: ", ".join(values))
    final_df["tags"] = final_df["tags"].apply(lambda values: " ".join(values).lower())

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(final_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return final_df, similarity


MOVIES_DF, SIMILARITY = _load_data()


def recommend(movie_title, limit=5):
    matches = MOVIES_DF[MOVIES_DF["title"].str.lower() == movie_title.lower()]
    if matches.empty:
        raise ValueError(f"Movie '{movie_title}' not found.")

    movie_index = matches.index[0]
    distances = list(enumerate(SIMILARITY[movie_index]))
    ranked_movies = sorted(distances, reverse=True, key=lambda item: item[1])[1 : limit + 1]

    return [MOVIES_DF.iloc[index].title for index, _ in ranked_movies]
