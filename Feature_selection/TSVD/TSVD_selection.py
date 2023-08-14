import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import TruncatedSVD

def TSVD(data,n_components=300):
    svd = TruncatedSVD(n_components=n_components)
    new_data=svd.fit_transform(data)  
    return new_data

data_input = pd.read_csv(r'RPI369_protein_PN_ALL.csv')
data_ = np.array(data_input)
data = data_[:,1:]
label = data_[:,0]
Zongshu = scale(data)
RNA_shu = Zongshu[:,0:32]
pro_shu = Zongshu[:,32:]

new_RNA_data = TSVD(RNA_shu,n_components=1)
new_pro_data = TSVD(pro_shu,n_components=67)

data_new = np.hstack((new_RNA_data,new_pro_data))
optimal_RPI_features = pd.DataFrame(data=data_new)
optimal_RPI_features.to_csv('RPI369_protein_TSVD.csv')


