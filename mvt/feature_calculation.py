# RDKit Feature Calculation

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.inchi import MolFromInchi

df = pd.read_csv("../data/RBioDeg.csv")

df = df[['CAS', 'InChI_Code_QSARr', 'Ready_Biodeg']]

nms = [x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
for i in range(len(df)):
    descrs = calc.CalcDescriptors(MolFromInchi(df.loc[i, 'InChI_Code_QSARr']))
    for x in range(len(descrs)):
        df.at[i, str(nms[x])] = descrs[x]
df = df.dropna()

df.to_csv('../data/RBioDeg_RDKit_Features.csv')
