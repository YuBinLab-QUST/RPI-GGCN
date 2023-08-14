# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:48:53 2023

@author: dell
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import KernelPCA, SparsePCA,PCA

def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)#axis represents the obtained mean value by column
    stdVal=np.std(dataMat,axis=0)
    newData=dataMat-meanVal
    new_data=newData/stdVal
    return new_data,meanVal
def covArray(dataMat):
    #obtain the  covariance matrix
    covMat=np.cov(dataMat,rowvar=0)
    return covMat
def featureMatrix(dataMat):
    eigVals,eigVects=np.linalg.eig(np.mat(dataMat))
	#datermine the eigenvalue and eigenvector
    return eigVals,eigVects
def percentage2n(eigVals,percentage=0.99):  
    #percentage represents the rate of contribution
    sortArray=np.sort(eigVals)   #ascending sort 
    sortArray=sortArray[-1::-1]  #descending order
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num #num is the number of remaining principal component


def pca(data,percentage = 0.9):  
    dataMat = np.array(data) 
    newData,meanVal=zeroMean(data)  #equalization
    covMat=covArray(newData)  #covariance matrix
    eigVals,eigVects=featureMatrix(covMat)
    n_components = percentage2n(eigVals,percentage)
    clf=PCA(n_components=n_components)  
    new_x = clf.fit_transform(dataMat)
    return new_x

data_input = pd.read_csv(r'RPI369_protein_PN_ALL.csv')
data_ = np.array(data_input)
data = data_[:,1:]
label = data_[:,0]
Zongshu = scale(data)
RNA_shu = Zongshu[:,0:6]
pro_shu = Zongshu[:,500:]

new_RNA_data = pca(RNA_shu,percentage=1)
new_pro_data = pca(pro_shu,percentage=12)

data_new = np.hstack((new_RNA_data,new_pro_data))
optimal_RPI_features = pd.DataFrame(data=data_new)
optimal_RPI_features.to_csv('RPI369_protein_PCA.csv')

