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
from sklearn.decomposition import PCA
from bikMeans import *

################################################################# 聚类模型 #################################################################
# 用畸变程度和轮廓系数可以评估聚类效果
# 可以考虑先降维
# K-Means参数的最优解也是以成本函数最小化为目标


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



# 整体判断，筛选出离群点，并基于0标签数据重新聚类
# bisecting k-means

datMat = np.mat(train_data)
bik_bst= bikMeans(datMat,2)


print "最终质心：\n",bik_bst[0]
print "索引值和均值：\n",bik_bst[1]




# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
# kmeans=MiniBatchKMeans(init='k-means++'
#                       ,n_clusters=2
#                       ,batch_size=100
#                       ,reassignment_ratio=5e-4
#                       ,max_no_improvement=10
#                       ,random_state=110)


kmeans=MiniBatchKMeans(init=centroids
                       ,n_clusters=2
                       ,batch_size=100
                       ,reassignment_ratio=5e-4
                       ,max_no_improvement=10
                       ,random_state=110)


outer_bst=kmeans.fit(train_data)  #分组
label_pred = outer_bst.labels_ # 每个数据点所属分组
centroids = outer_bst.cluster_centers_ #获取分组中心

# 异常点数量
sum(list(label_pred)) #36190


# list(label_pred).count(0)
# list(label_pred).count(1)

# 输出聚类标签，并合并数据 : [wifi_list，聚类标签]
wifi_lable=pd.DataFrame(list(outer_bst.labels_),columns=['outer_label'])
data_set=pd.concat([train_data,wifi_list_data,wifi_lable],axis=1)


# 筛选异常点
outer_data=data_set[data_set.outer_label==1]

outer_data[['wifi_list','outer_label']].to_csv(xpath+'/outer_data.txt',sep = '\t',header = False,index = False,encoding='utf-8')


# 新加入数据进行预测
# outer_Label=outer_bst.predict(newData)




#剔除异常点生成新的数据集，并拆分数据
data_set_result=data_set[data_set.outer_label!=1]
train_data=data_set_result.drop(['wifi_list','outer_label'],axis=1)
wifi_list_data=data_set_result['wifi_list']


################################################################# PCA #################################################################

n=9
pca=PCA(n_components=n,copy=True,whiten=True)  #返回所保留的n个主成分个数
pca.fit(train_data)

# 对象的属性值
pca.explained_variance_
pca.explained_variance_ratio_  #返回所保留的n个主成分各自的方差百分比
pca.singular_values_

# 对应数据A，用已训练好的pca模型进行降维
PCA_TrainData=pca.fit_transform(train_data)
new_TrainData=pd.DataFrame(PCA_TrainData,columns=[chr(i) for i in range(97,97+n)])



################################################################# 挑选最优的K值 #################################################################
# 肘部法则：随着聚类中心个数 K 的增加，其代价函数计算的结果会下降，不是选择最小的代价函数，而是选择突变点
def Elbows_Rule(file):
    elbow_coefs=[]
    for k in range(2,20):
        kmeans=KMeans(n_clusters=k,max_iter=100)
        bst=kmeans.fit(file)
        elbow_coef=sum(np.min(cdist(file,kmeans.cluster_centers_,'euclidean'),axis=1))/file.shape[0]
        elbow_coefs.append(elbow_coef)
    coefs_sub=[elbow_coefs[i]-elbow_coefs[i+1] for i in range(len(elbow_coefs)-1)]
    elbow_coef=elbow_coefs[coefs_sub.index(max(coefs_sub))+1]
    index=elbow_coefs.index(elbow_coef)+1
    return coefs_sub,elbow_coef,index



# 最优轮廓系数，选择k值
Elbows_result=Elbows_Rule(new_TrainData)
Elbows_coef=Elbows_result[1]
Elbows_index=Elbows_result[2]



# 轮廓系数：选择 max(scores)
# 方法1：Silhouette Coefficient
def Silhouette_score(file):
    scores=[]
    for k in range(2,20):
        kmeans=KMeans(n_clusters=k)
        bst=kmeans.fit(file)
        score=metrics.silhouette_score(file,kmeans.labels_,metric='euclidean')
        scores.append(score)
    score_sub=[scores[i]-scores[i+1] for i in range(len(scores)-1)]
    score=scores[score_sub.index(max(score_sub))+1]
    index=scores.index(score)+1
    return score_sub,score,index


# 最优轮廓系数 max(Silhouette_score)
Sil_result=Silhouette_score(new_TrainData)
Sil_score=Sil_result[1]
Sil_index=Sil_result[2]



# 方法1：Calinski-Harabaz Index，即类别内部数据的协方差越小越好，类别之间的协方差越大越好
def Calinski_harabaz(file):
    scores=[]
    for k in range(2,20):
        kmeans=KMeans(n_clusters=k)
        bst=kmeans.fit(file)
        score=metrics.calinski_harabaz_score(file, kmeans.labels_)
        scores.append(score)
    score_sub=[scores[i]-scores[i+1] for i in range(len(scores)-1)]
    score=scores[score_sub.index(max(score_sub))+1]
    index=scores.index(score)+1
    return score_sub,score,index


# 计算轮廓系数 max(Silhouette_score)
Cali_result=Calinski_harabaz(new_TrainData)
Cali_score=Cali_result[1]
Cali_index=Cali_result[2]



########################################################## 根据最优K值重新跑模型 #########################################################
# 获取聚类准则的最后值
kmeans=KMeans( n_clusters=Elbows_index
              ,init='k-means++'
              ,n_init=10 #获取初始化簇心的更迭次数
              ,max_iter=300 # 最大迭代次数
              ,tol=0.001 # 运行准则收敛的条件
              ,algorithm='auto'  # kmeans的实现算法
              )



bst=kmeans.fit(new_TrainData)  #分组
label_pred = bst.labels_ # 每个数据点所属分组
centroids = bst.cluster_centers_ #获取分组中心
inertia = bst.inertia_   # 获取分组准则的最后值



# 输出聚类标签，并合并数据 : [wifi_list，聚类标签]
wifi_lable=pd.DataFrame(list(bst.labels_),columns=['label'])
result_1=pd.concat([wifi_list_data,wifi_lable],axis=1)


# 判断每一类样本数量
a={}
for i in range(Elbows_index):
    a[i]=result_1[result_1.label==i].shape[0]


# 抽样2
result_1[result_1.wifi_list=='08:10:79:9e:ce:5d']
data_set[data_set.wifi_list=='00:03:7f:01:02:03']


########################################################## 根据新数据判断属于哪一类 #########################################################
#http://blog.topspeedsnail.com/archives/10349

# 新数据
new_data=data_set.drop(['wifi_list'],axis=1).iloc[10000:10003,:]

# 对于新数据首先判断是否outer，如果不是
outer_label=outer_bst.predict(newData)


new_data_result=new_data[new_data.new_label!=1]
train_data=new_data.drop(['wifi_list'],axis=1)
wifi_list_data=new_data['wifi_list']


#利用训练好的PCA模型生成新的数据集并转化成数据框的形式
newA=pca.transform(new_data_result)
newData=pd.DataFrame(newA,columns=[chr(i) for i in range(97,97+n)])


# 结合pca模型和bst模型识别新的数据集属于哪一类

newLabel=bst.predict(newData)

# new_data的label即为outer_label和newLabel的组合



###############################################################################


# 对所有数据进行聚类，包括离群点，生成模型A
# 删除所有标识离群点label的样本，重新生成聚类模型B
# 基于聚类模型A，predict新的样本点，如果数据离群点类，则打上离群label
# 对于未打上离群label的样本用模型B重新分组
# 结合A模型和B模型的label即为样本数据的所有label



##############

x=np.array([[1,2],[2,3],[1.5,1.8],[1,0.6],[1,2],[2,3],[1.5,1.8],[1,0.6],[10,11],[11,15],[1.1,1.6]])
KMeans(n_clusters=2).fit(x).labels_


clf=MiniBatchKMeans(n_clusters=2,batch_size=4)
bst=clf.fit(x)
bst.labels_
predict=[[1.6,1.9]]
bst.predict(predict)

#############




# http://lilian.info/blog/2016/12/sklearn.html
######## preprocessing
# StandardScaler
# MaxAbsScaler
# MinMaxScaler
# RobustScaler
# Normalizer