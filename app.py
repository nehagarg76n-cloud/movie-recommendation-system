import os
from urllib.parse import quote_plus

import requests
import streamlit as st

from recommender import MOVIES_DF, recommend


TMDB_API_KEY = os.getenv("TMDB_API_KEY", "16e66e13e21bbb662d68f616461d7ecd")
PLACEHOLDER_POSTER = "https://via.placeholder.com/300x450?text=No+Image"


def fetch_poster(movie_name):
    if not TMDB_API_KEY:
        return PLACEHOLDER_POSTER

    search_url = (
        "https://api.themoviedb.org/3/search/movie"
        f"?api_key={TMDB_API_KEY}&query={quote_plus(movie_name)}"
    )

    try:
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return PLACEHOLDER_POSTER

    results = data.get("results", [])
    if results:
        poster_path = results[0].get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path.lstrip('/')}"

    return PLACEHOLDER_POSTER


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    .title {text-align:center; font-size:50px; font-weight:bold;}
    .subtitle {text-align:center; color:gray;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🎬 Movie Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Search & discover movies like Netflix</div>',
    unsafe_allow_html=True,
)

search_query = st.text_input("🔍 Search movie")

genres = ["All"] + sorted(MOVIES_DF["genres"].dropna().unique().tolist())
selected_genre = st.selectbox("🎭 Select Genre", genres)

filtered_movies = MOVIES_DF.copy()

if search_query:
    filtered_movies = filtered_movies[
        filtered_movies["title"].str.contains(search_query, case=False, na=False)
    ]

if selected_genre != "All":
    filtered_movies = filtered_movies[
        filtered_movies["genres"].str.contains(selected_genre, case=False, na=False)
    ]

movie_list = sorted(filtered_movies["title"].tolist())
movie_name = None

if movie_list:
    movie_name = st.selectbox("🎥 Select a movie", movie_list)
else:
    st.warning("No movies matched your search/filter.")


if st.button("Recommend Movies") and movie_name:
    results = recommend(movie_name, limit=7)

    st.markdown("## 🍿 Top Recommendations")
    cols = st.columns(7)

    for idx, movie in enumerate(results[:7]):
        poster = fetch_poster(movie)

        with cols[idx]:
            st.image(poster)
            st.caption(movie)
