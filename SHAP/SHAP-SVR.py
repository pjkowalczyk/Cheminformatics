import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import inchi
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.ML.Descriptors import MoleculeDescriptors

from pprint import pprint

df = pd.read_csv('data/activity_classes_ChEMBL24.dat', sep='\t')

H3 = df[df.TargetName.eq('Histamine H3 receptor')]
H3.head(1)

H3.hist('pKi', bins = 50)

PandasTools.AddMoleculeColumnToFrame(H3, smilesCol = "SMILES")
H3.head(1)

radius = 2
nBits = 1024

ECFP4 = [AllChem.GetMorganFingerprintAsBitVect(x, radius = radius, nBits = nBits) for x in H3['ROMol']]

ecfp4_names = [f'Bit_{i}' for i in range(nBits)]
ecfp4_bits = [list(l) for l in ECFP4]
H3_ecfp4 = pd.DataFrame(ecfp4_bits, index = H3.SMILES, columns = ecfp4_names)
H3_ecfp4.head(1)

H3_pKi = H3[['SMILES', 'pKi']]
H3_pKi.set_index('SMILES', inplace = True)

H3_Xy = H3_pKi.merge(H3_ecfp4, on = 'SMILES')

X_train, X_test, y_train, y_test = train_test_split(H3_Xy.drop(['pKi'], axis = 1), H3_Xy['pKi'],
                                                   test_size = 0.2,
                                                   random_state = 42)

# create dataframe, select columns
df1x = pd.to_numeric(pd.DataFrame(y_train)['pKi'])
df2x = pd.to_numeric(pd.DataFrame(y_test)['pKi'])
#Stack the data
plt.figure()
plt.hist([df1x,df2x], bins = 20, stacked = False, density = True)
plt.title('pKi Distributions', fontsize = 16)
plt.xlabel('pKi', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(['train', 'test'])
plt.show()

SVR_model = SVR()

SVR_model.fit(X_train, y_train)

y_pred = SVR_model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
rmse

y_pred = SVR_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse

import shap
shap.initjs()

explainer = shap.KernelExplainer(SVR_model.predict, shap.sample(X_test, 5))
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type = 'bar')

shap.summary_plot(shap_values, X_test)

# Get the predictions and put them in with the test data
X_output = X_test.copy()
X_output.loc[:, 'predict'] = np.round(SVR_model.predict(X_output), 2)

# Randomly pick some observations
random_picks = np.arange(1, 193, 50)
qaz = X_output.iloc[random_picks]
qaz

m = Chem.MolFromSmiles('Cc1c(c(no1)c2ccccc2)c3ccc4cc(ccc4n3)CCN5CCCC5C')
m

bi = {}
fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, nBits = 1024, bitInfo=bi)
# show 10 of the set bits:
list(fp.GetOnBits())

def shap_plot(j):
#    explainerModel = shap.KernelExplainer(SVR_model, X)
    shap_values_Model = explainer.shap_values(qaz.drop(['predict'], axis = 1))
    p = shap.force_plot(explainer.expected_value, shap_values_Model[j], qaz.drop(['predict'], axis = 1).iloc[[j]])
    return(p)

shap_plot(3)

