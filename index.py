from pathlib import Path
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import nltk


BASE_DIR = Path(__file__).resolve().parent

credits_path = BASE_DIR / "tmdb_5000_credits.csv" / "tmdb_5000_credits.csv"
movies_path = BASE_DIR / "tmdb_5000_movies.csv" / "tmdb_5000_movies.csv"

if not credits_path.exists():
    credits_path = BASE_DIR / "tmdb_5000_credits.csv"

if not movies_path.exists():
    movies_path = BASE_DIR / "tmdb_5000_movies.csv"

credit = pd.read_csv(credits_path)
movie = pd.read_csv(movies_path)
movies = movie.merge(credit, on="title")
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

movies.dropna(inplace=True)
movies.reset_index(drop=True, inplace=True)


def convert(obj):
    names = []
    for item in literal_eval(obj):
        names.append(item["name"])
    return names


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)

def convert3(obj):
    names = []
    counter=0
    for item in literal_eval(obj):
        if counter!=3:
            names.append(item["name"])
            counter+=1
        else:
            break
    return names            
        
        
   
movies["cast"] = movies["cast"].apply(convert3)

def fetch_director(obj):
    names = []
    for item in literal_eval(obj):
        if item['job']=='Director':
            names.append(item["name"])
            break
          
        
    return names
movies["crew"] = movies["crew"].apply(fetch_director)

movies["crew"].head()

movies['overview']=movies['overview'].apply(lambda x:x.split())


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags']=movies['overview']+movies['genres'] + movies['keywords'] +movies['cast']+ movies['crew']
new_df = movies[["movie_id", "title", "tags"]].copy()

#conver tags list into string 
new_df['tags']= new_df['tags'].apply(lambda x:" ".join(x))

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words="english")
vectors=cv.fit_transform(new_df["tags"]).toarray()

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()



def stem(text):
    l=[]
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)

print(ps.stem('loving'))    
    
cv.get_feature_names_out()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:8]

def recommend(movie):
    matches = new_df[new_df["title"].str.lower() == movie.lower()]
    if matches.empty:
        raise ValueError(f"Movie '{movie}' not found.")

    movie_index = matches.index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:8]
    return [new_df.iloc[i[0]].title for i in movies_list]


for title in recommend("avatar"):
    print(title)
