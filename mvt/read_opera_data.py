import pandas as pd
from rdkit.Chem import PandasTools

### Read data
train_df = PandasTools.LoadSDF("../OPERA_Data_SDF/TR_RBioDeg_1197.sdf")
test_df = PandasTools.LoadSDF("../OPERA_Data_SDF/TST_RBioDeg_411.sdf")

### Concatenate data
### Note: The target, 'LogHalfLife' in this instance, is specific for each dataset.
### The script needs to be edited to reflect the target identifier. 
RBioDeg = pd.concat([train_df[["CAS", "Canonical_QSARr", "InChI_Code_QSARr", "InChI Key_QSARr", "Ready_Biodeg"]],
                     test_df[["CAS", "Canonical_QSARr", "InChI_Code_QSARr", "InChI Key_QSARr", "Ready_Biodeg"]]], ignore_index = True)

RBioDeg['Ready_Biodeg'] = [float(x) for x in RBioDeg.Ready_Biodeg]

RBioDeg.to_csv("../data/RBioDeg.csv")
