from sklearn.mixture import GaussianMixture
from app.preprocessing.preprocessing import preprocessing , PCA_transform
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df = pd.read_csv("../cross_genre_music_discovery/raw_data/spotify-tracks-dataset-detailed.csv")

df = preprocessing(df)

df = PCA_transform(df)

print(df)
