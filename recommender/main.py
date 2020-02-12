import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load ratings
#df = pd.read_csv('./ml-latest-small/ratings.csv', sep=',', names=['userId','movieId','rating','timestamp'])
df = pd.merge(df, movie_titles, on='item_id')

# Load movies
movie_titles = pd.read_csv('./ml-latest-small/ratings.csv')

df = pd.merge(df, movie_titles, on='movieId')
