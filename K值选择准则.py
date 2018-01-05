#!/usr/bin/python
#-*- coding:utF-8 -*-

import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from sklearn import metrics


################################################################# 聚类模型 #################################################################
# 用畸变程度和轮廓系数可以评估聚类效果
# 可以考虑先降维
# K-Means参数的最优解也是以成本函数最小化为目标
# 为了避免没一类总样本太少，可以用MiniBatchKMeans替代kmeans

################################################################# load_data #################################################################
xpath='/home/fsg/wenchao_model/wifi'

#1、设置异常值替换标准
na_values=['','NULL','NA','null','na','Na','-9999','-1','Infinity','NaN']

xfile_name='/wifi_name.txt'
#2、读取列名
data_set_name=pd.read_table(xpath+xfile_name,
                         header=None,
                         sep='\t',
                         names=['var_name']
                         )


xfile_data='/wifi_data_set.txt'
#3、加载数据
data_set=pd.read_table(xpath+xfile_data,
                         sep='\t',
                         header=None,
                         na_values=na_values,
                         names=data_set_name['var_name'].values
                         )


# 拆分数据
train_data=data_set.drop(['wifi_list'],axis=1)
wifi_list_data=data_set['wifi_list']

train_data=train_data.iloc[0:10000,:]
################################################################# 挑选最优的K值 #################################################################
# 肘部法则：随着聚类中心个数 K 的增加，其代价函数计算的结果会下降，不是选择最小的代价函数，而是选择突变点
def Elbows_Rule(file):
    elbow_coefs=[]
    for k in range(2,20):
        kmeans=MiniBatchKMeans(n_clusters=4,batch_size=300,reassignment_ratio=5e-4,max_no_improvement=10)
        bst=kmeans.fit(file)
        elbow_coef=sum(np.min(cdist(file,kmeans.cluster_centers_,'euclidean'),axis=1))/file.shape[0]
        elbow_coefs.append(elbow_coef)
    coefs_sub=[elbow_coefs[i]-elbow_coefs[i+1] for i in range(len(elbow_coefs)-1)]
    elbow_coef=elbow_coefs[coefs_sub.index(max(coefs_sub))+1]
    index=elbow_coefs.index(elbow_coef)+1
    return coefs_sub,elbow_coef,index



# 最优轮廓系数
Elbows_result=Elbows_Rule(train_data)
Elbows_coef=Elbows_result[1]
Elbows_index=Elbows_result[2]



# 轮廓系数：选择 max(scores)
# 方法1：Silhouette Coefficient
def Silhouette_score(file):
    scores=[]
    for k in range(2,20):
        kmeans=MiniBatchKMeans(n_clusters=4,batch_size=300,reassignment_ratio=5e-4,max_no_improvement=10)
        bst=kmeans.fit(file)
        score=metrics.silhouette_score(file,kmeans.labels_,metric='euclidean')
        scores.append(score)
    score_sub=[scores[i]-scores[i+1] for i in range(len(scores)-1)]
    score=scores[score_sub.index(max(score_sub))+1]
    index=scores.index(score)+1
    return score_sub,score,index


# 最优轮廓系数 max(Silhouette_score)
Sil_result=Silhouette_score(train_data)
Sil_score=Sil_result[1]
Sil_index=Sil_result[2]



# 方法1：Calinski-Harabaz Index，即类别内部数据的协方差越小越好，类别之间的协方差越大越好
def Calinski_harabaz(file):
    scores=[]
    for k in range(2,20):
        kmeans=MiniBatchKMeans(n_clusters=4,batch_size=300,reassignment_ratio=5e-4,max_no_improvement=10)
        bst=kmeans.fit(file)
        score=metrics.calinski_harabaz_score(file, kmeans.labels_)
        scores.append(score)
    score_sub=[scores[i]-scores[i+1] for i in range(len(scores)-1)]
    score=scores[score_sub.index(max(score_sub))+1]
    index=scores.index(score)+1
    return score_sub,score,index


# 计算轮廓系数 max(Silhouette_score)
Cali_result=Calinski_harabaz(train_data)
Cali_score=Cali_result[1]
Cali_index=Cali_result[2]



##### 轮廓系数 和 肘部法则 选择最优的K值
