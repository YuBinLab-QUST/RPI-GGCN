import argparse
import configparser
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import time
import sys
import os


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from model import*
from utils import*

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
import pandas as pd
import random
from sklearn.metrics import roc_auc_score
import math
import os
'''
random.seed(2)
np.random.seed(2)
os.environ['PYTHONHASHSEED'] = str(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''


def to_torch_sparse_tensor(x, device='cpu'):
    if not sp.isspmatrix_coo(x):
        x = sp.coo_matrix(x)
    row, col = x.row, x.col
    data = x.data

    indices = torch.from_numpy(np.asarray([row, col]).astype('int64')).long()
    values = torch.from_numpy(data.astype(np.float32))
    th_sparse_tensor = torch.sparse.FloatTensor(indices, values,
                                                x.shape).to(device)

    return th_sparse_tensor


def tensor_from_numpy(x, device='cpu'):
    return torch.from_numpy(x).to(device)


def globally_normalize_bipartite_adjacency(adjacencies, symmetric=False):
    """ 传入的是邻接矩阵的集合，要在不同的关系矩阵上分别计算 """

    row_sum = [np.sum(adj, axis=1) for adj in adjacencies]
    col_sum = [np.sum(adj, axis=0) for adj in adjacencies]
    # 将为0的设置为无穷大，避免除0
    for i in range(len(row_sum)):
        row_sum[i][row_sum[i] == 0] = np.inf
        col_sum[i][col_sum[i] == 0] = np.inf
    degree_row_inv = [1./r for r in row_sum]
    degree_row_inv_sqrt = [1./np.sqrt(r) for r in row_sum]
    degree_col_inv_sqrt = [1./np.sqrt(c) for c in col_sum]
    normalized_adj = []
    if symmetric:
        for i, adj in enumerate(adjacencies):
            normalized_adj.append(np.diag(degree_row_inv_sqrt[i]).dot(adj).dot(np.diag(degree_col_inv_sqrt[i])))
    else:
        for i, adj in enumerate(adjacencies):
            normalized_adj.append(np.diag(degree_row_inv[i]).dot(adj))
    return normalized_adj


def get_k_fold_data(k, data):
    data = data.values
    X, y = data[:, :], data[:, -1]
    #sfolder = StratifiedKFold(n_splits = k, shuffle=True,random_state=1)
    sfolder = StratifiedKFold(n_splits=k, shuffle=True)

    train_data = []
    test_data = []
    train_label = []
    test_label = []

    for train, test in sfolder.split(X, y):
        train_data.append(X[train])
        test_data.append(X[test])
        train_label.append(y[train])
        test_label.append(y[test])

        # print('Train: %s | test: %s' % (X[train], X[test]))
        # print('label:%s|label:%s' % (y[train], y[test]))
        #print(len(test))
        #print(X[train][:-1,:])
    return train_data, test_data


def AUC(label, prob):
    return roc_auc_score(label, prob)


def true_positive(pred, target):
    return ((pred == 1) & (target == 1)).sum().clone().detach().requires_grad_(False)


def true_negative(pred, target):
    return ((pred == 0) & (target == 0)).sum().clone().detach().requires_grad_(False)


def false_positive(pred, target):
    return ((pred == 1) & (target == 0)).sum().clone().detach().requires_grad_(False)


def false_negative(pred, target):
    return ((pred == 0) & (target == 1)).sum().clone().detach().requires_grad_(False)


def precision(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def sensitivity(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out

def specificity(pred, target):
    tn = true_negative(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)

    out = tn/(tn+fp)
    out[torch.isnan(out)] = 0

    return out


def MCC(pred,target):
    tp = true_positive(pred, target).to(torch.float)
    tn = true_negative(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)

    out = (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tn+fn)*(tp+fn)*(tn+fp))
    out[torch.isnan(out)] = 0

    return out

def accuracy(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    tn = true_negative(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)
    out = (tp+tn)/(tp+tn+fn+fp)
    out[torch.isnan(out)] = 0

    return out


def FPR(pred, target):
    fp = false_positive(pred, target).to(torch.float)
    tn = true_negative(pred, target).to(torch.float)
    out = fp/(fp+tn)
    out[torch.isnan(out)] = 0
    return out


def TPR(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)
    out = tp/(tp+fn)
    out[torch.isnan(out)] = 0
    return out


def printN(pred, target):
    TP = true_positive(pred, target)
    TN = true_negative(pred, target)
    FP = false_positive(pred, target)
    FN = false_negative(pred, target)
    print("TN:{},TP:{},FP:{},FN:{}".format(TN, TP, FP, FN))
    return TP,TN,FP,FN


def performance(tp,tn,fp,fn):
    final_tp = 0
    final_tn = 0
    final_fp = 0
    final_fn = 0
    for i in range(len(tp)):
        final_fn += fn[i]
        final_fp += fp[i]
        final_tn += tn[i]
        final_tp += tp[i]
    print("TN:{},TP:{},FP:{},FN:{}".format(final_tn, final_tp, final_fp, final_fn))
    ACC = (final_tp + final_tn) /float (final_tp + final_tn + final_fn + final_fp)
    Sen = final_tp / float(final_tp+ final_fn)
    Spe = final_tn/float(final_tn+final_fp)
    Pre = final_tp / float(final_tp + final_fp)
    MCC = (final_tp*final_tn-final_fp*final_fn)/float(math.sqrt((final_tp+final_fp)*(final_tn+final_fn)*(final_tp+final_fn)*(final_tn+final_fp)))
    FPR = final_fp/float(final_fp+final_tn)
    return ACC,Sen, Spe,Pre,MCC,FPR

# ==================================================


DEVICE = torch.device('cpu')
SCORES = torch.tensor([-1, 1]).to(DEVICE)
'''
seed = 1
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''


def load_dataset(dataset, filepath, identity_feature, negative_random_sample, use_side_feature, identity_feature_dim=1024):
    print(use_side_feature)
    filepath = os.path.join(filepath,dataset)
    NPI_pos_matrix = pd.read_csv(os.path.join(filepath,'NPI_pos.csv'), header=None).values

    name = ['index']
    for i in range(1024):
        name.append(i + 1)

    NPI_neg_matrix = pd.read_csv(os.path.join(filepath,"NPI_neg_" + negative_random_sample + ".csv"), header=None).values
    edgelist = pd.read_csv(os.path.join(filepath,'edgelist_' + negative_random_sample + '.csv'), header=None)

    if use_side_feature:
        protein_side_feature = pd.read_csv( os.path.join(filepath, 'Protein.csv')).values
        RNA_side_feature = pd.read_csv(os.path.join(filepath,'ncRNA.csv')).values
        supplement = np.zeros((RNA_side_feature.shape[0], 87))  # 通过补零补齐到同一维度
        RNA_side_feature = np.concatenate((RNA_side_feature, supplement), axis=1)
    else:
        protein_side_feature = []
        RNA_side_feature = []

    
data_start = ('NPI_pos_matrix.csv'+'side_feature.csv')
label_P = np.ones(int('243'))
label_N = np.zeros(int('245'))
label_start = np.hstack((label_P,label_N))
label=np.array(label_start)
data=np.array(data_start)
shu=scale(data)
y= label

[sample_num,input_dim]=np.shape(shu)
X = np.reshape(shu,(-1,1,input_dim))
out_dim=2

ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
sepscores = []
sepscores_ = []


  
def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp + 1e-06)
    npv = float(tn)/(tn + fn + 1e-06)
    sensitivity = float(tp)/ (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)
    mcc = float(tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-06)
    f1=float(tp*2)/(tp*2+fp+fn+1e-06)
    return acc, precision,npv, sensitivity, specificity, mcc, f1

skf= StratifiedKFold(n_splits=5)

for train, test in skf.split(X,y): 
    y_train=to_categorical(y[train])#generate the resonable results
    cv_clf = model(input_dim,out_dim)
    hist=cv_clf.fit(X[train], 
                    y_train,
                    epochs=30)
    y_test=to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]       
    y_score=cv_clf.predict(X[test])#the output of  probability
    yscore=np.vstack((yscore,y_score))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= categorical_probas_to_classes(y_score)
    acc, precision,npv, sensitivity, specificity, mcc,f1 = calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('Results: acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
    hist=[]
    cv_clf=[]
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))


   