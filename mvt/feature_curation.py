import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/RBioDeg_RDKit_Features.csv')

df = df.drop(columns = ['Unnamed: 0', 'InChI_Code_QSARr'])

df = df.loc[:, ~df.columns.str.startswith('fr_')]

df = df.loc[:, ~df.columns.str.startswith('Num')]

df.set_index('CAS', inplace = True)

X = df.drop(columns = ['Ready_Biodeg'])
y = df['Ready_Biodeg']

### Identify / remove near-zero variance descriptors
def variance_threshold_selector(data, threshold = 0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices = True)]]

nzv = variance_threshold_selector(X, 0.0)

X = X[nzv.columns]

### Identify / remove highly correlated descriptors
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                  k = 1).astype(np.bool))
to_drop = [column for column in upper.columns
           if any(upper[column] > 0.85)]

X = X[X.columns.drop(to_drop)]

### standardize features by removing the mean and scaling to unit variance
scaled_features = StandardScaler().fit_transform(X.values)
scaled_features_X = pd.DataFrame(scaled_features, index = X.index, columns = X.columns)

y = pd.DataFrame(y)

curated_df = y.join(scaled_features_X)

curated_df.to_csv("../data/RBioDeg_curated.csv")
