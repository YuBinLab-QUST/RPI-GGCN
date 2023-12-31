import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.linear_model import LassoCV

def lassodimension2(data,label,alpha=np.array([0.01])):#alpha代表想要传递的alpha的一组值,用在循环中,来找出一个尽可能好的alpha的值
    lassocv=LassoCV(cv=5, alphas=alpha,max_iter=500).fit(data, label)
    x_lasso = lassocv.fit(data,label)#代入alpha进行降维
    mask = x_lasso.coef_ != 0 #mask是一个numpy数组,数组中的元素都是bool值,并且数组的维度和data的维度是相同的
    new_data = data[:,mask]  #将data中相应维度中与mask中为True对应的元素挑选出来
    return new_data,mask #返回降维之后的数组,并将使用lasso训练数据之后得到的最大值一起返回

data_input=pd.read_csv(r'RPI369_protein_PN_ALL.csv')
data_=np.array(data_input)
data=data_[:,1:]
label=data_[:,0]
Zongshu=scale(data)
RNA_shu = Zongshu[:,0:256]
pro_shu = Zongshu[:,256:]

new_RNA_data,index_RNA=lassodimension2(RNA_shu,label,alpha=np.array([8.4]))
new_pro_data,index_pro=lassodimension2(pro_shu,label,alpha=np.array([11.7]))

data_new = np.hstack((new_RNA_data,new_pro_data))
optimal_RPI_features= pd.DataFrame(data=data_new)
optimal_RPI_features.to_csv('RPI369_protein_LASSO.csv')



