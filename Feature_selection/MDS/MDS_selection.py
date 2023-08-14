import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.manifold import MDS

def MDS_select(data,n_components=300):
    embedding = MDS(n_components=n_components)
    new_data = embedding.fit_transform(data)
    return new_data

data_input = pd.read_csv(r'RPI369_rna_PN_ALL.csv')
data_ = np.array(data_input)
data = data_[:,1:]
label = data_[:,0]
Zongshu = scale(data)
RNA_shu = Zongshu[:,0:72]
pro_shu = Zongshu[:,72:]

new_RNA_data = MDS_select(RNA_shu,n_components=7)
new_pro_data = MDS_select(pro_shu,n_components=30)

data_new = np.hstack((new_RNA_data,new_pro_data))
optimal_RPI_features = pd.DataFrame(data=data_new)
optimal_RPI_features.to_csv('RPI369_RNA_MDS.csv')
