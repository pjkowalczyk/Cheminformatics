import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px

df = pd.read_csv('../data/BioHL_curated.csv')

df.set_index('CAS', inplace = True)

X = df.drop(columns = ['LogHalfLife'])
y = df['LogHalfLife']

qaz = df.sort_values(by=['LogHalfLife'], ascending=False)

neighbors = euclidean_distances(X, pd.DataFrame(qaz.iloc[0, ]))

# Euclidean distances between rows of X
X_dist = euclidean_distances(X, X)

y_dist = pairwise_distances(pd.DataFrame(y), metric = 'euclidean')

chemistry_dist = list(X_dist[np.triu_indices(n = len(X_dist), k = 1)])
target_dist = list(y_dist[np.triu_indices(n = len(y_dist), k = 1)])

dist = pd.DataFrame(list(zip(chemistry_dist, target_dist)), 
               columns =['Feature_Distance', 'Target_Distance']) 
dist.sample(5).head() 

dist.plot(kind='hexbin', x = 'Feature_Distance', y = 'Target_Distance',
          gridsize = 25)

fig = px.box(dist, y="Feature_Distance", points="all")
fig.show()

cuts = pd.qcut(dist['Feature_Distance'], 20, retbins = True)
