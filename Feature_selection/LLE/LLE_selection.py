import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.manifold import LocallyLinearEmbedding

def LLE(data,n_components=300):
    embedding = LocallyLinearEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed

data_input = pd.read_csv(r'RPI369_protein_PN_ALL.csv')
data_ = np.array(data_input)
data = data_[:,1:]
label = data_[:,0]
Zongshu = scale(data)
RNA_shu = Zongshu[:,0:500]
pro_shu = Zongshu[:,500:]

new_RNA_data = LLE(RNA_shu,n_components=33)
new_pro_data = LLE(pro_shu,n_components=35)

data_new = np.hstack((new_RNA_data,new_pro_data))
optimal_RPI_features = pd.DataFrame(data=data_new)
optimal_RPI_features.to_csv('RPI369_protein_LLE.csv')
