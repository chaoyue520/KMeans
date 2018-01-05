#encoding:utf-8
from numpy import *

def loadDataSet(filename):
    dataMat = []          #创建元祖
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = map(float,curLine) #使用map函数将curLine里的数全部转换为float型
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA,vecB):          #计算两个向量的欧式距离
    return sqrt(sum(power(vecA-vecB,2)))


def randCent(dataSet,k):            #位给定数据集构建一个包含k个随机质心的集合
    n = shape(dataSet)[1]   #shape函数此时返回的是dataSet元祖的列数
    centroids = mat(zeros((k,n)))       #mat函数创建k行n列的矩阵，centroids存放簇中心
    for j in range(n):
        minJ = min(dataSet[:,j])           #第j列的最小值
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)  #random.rand(k,1)产生shape(k,1)的矩阵
    return centroids



def kMeans(dataSet, k , disMeas = distEclud,createCent = randCent):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True
    ## step 1: init centroids
    centroids = createCent(dataSet, k)
    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in xrange(numSamples):
            minDist  = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = disMeas(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist  = distance
                    minIndex = j
             ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2
         ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis = 0)
    print 'Congratulations, cluster using k-means complete!'
    return centroids, clusterAssment


def bikMeans(dataSet,k,disMeas = distEclud):
    m = shape(dataSet)[0] #shape函数此时返回的是dataSet元祖的行数
    clusterAssment = mat(zeros((m,2)))      #创建一个m行2列的矩阵，第一列存放索引值，第二列存放误差，误差用来评价聚类效果
    #创建一个初始簇
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    print centList
    print len(centList)
    for j in range(m):
        clusterAssment[j,1] = disMeas(mat(centroid0),dataSet[j,:])**2 #计算所有点的均值，选项axis=0表示沿矩阵的列方向进行均值计算
    while (len(centList)<k):
        lowestSSE = inf #inf正无穷大
        for i in range(len(centList)):
            #尝试划分每一簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,disMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit and notSplit:",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit)<lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #更新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print "the bestCentToSplit is :",bestCentToSplit
        print "the len of bestClustAss is:",len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] =bestClustAss
    return centList,clusterAssment


datMat = mat(loadDataSet('a.txt'))
myCentList,myNewAssment = bikMeans(datMat,2)


print "最终质心：\n",myCentList
print "索引值和均值：\n",myNewAssment








