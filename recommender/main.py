
# Followed https://towardsdatascience.com/how-to-build-a-simple-recommender-system-in-python-375093c3fb7d
import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load ratings
#df = pd.read_csv('./ml-latest-small/ratings.csv', sep=',', names=['userId','movieId','rating','timestamp'])
df = pd.read_csv('./ml-latest-small/ratings.csv')

# Load movies
movie_titles = pd.read_csv('./ml-latest-small/movies.csv')

# Merge movie titles with ratings
df = pd.merge(df, movie_titles, on='movieId')

# Create a df of every moving with their rating
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

# Number of rating per movie
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()