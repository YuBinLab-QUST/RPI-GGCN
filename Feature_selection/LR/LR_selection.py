import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def LR_jiangwei(data,label,parameter=0.1):
    logistic_=LogisticRegression(penalty="l2", C=parameter)
    model=SelectFromModel(logistic_)
    new_data=model.fit_transform(data, label)
    mask=model.get_support(indices=True)
    return new_data,mask

data_input = pd.read_csv(r'RPI369_RNA_PN_ALL.csv')
data_ = np.array(data_input)
data = data_[:,1:]
label = data_[:,0]
Zongshu = scale(data)
RNA_shu = Zongshu[:,0:72]
pro_shu = Zongshu[:,72:]

new_RNA_data, index_RNA = LR_jiangwei(RNA_shu,label,parameter=0.8)
new_pro_data, index_pro = LR_jiangwei(pro_shu,label,parameter=0.8)

feature_numbe_RNA = ~index_RNA
H_RNA = np.argsort(feature_numbe_RNA)
mask_RNA = H_RNA[:7]
new_RNA_test_data = RNA_shu[:,mask_RNA]

feature_numbe_pro = ~index_pro
H_pro = np.argsort(feature_numbe_pro)
mask_pro = H_pro[:30]
new_pro_test_data = pro_shu[:,mask_pro]

data_new = np.hstack((new_RNA_test_data,new_pro_test_data))
optimal_RPI_features = pd.DataFrame(data=data_new)
optimal_RPI_features.to_csv('RPI369_RNA_LR.csv')
