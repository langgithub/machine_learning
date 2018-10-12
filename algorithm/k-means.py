"""
@version: 2.0.0
@author: lang
@license: Apache Licence 
@file: k-means.py
@time: 2018/8/17 15:28

                       .::::.
                     .::::::::.
                    :::::::::::
                ..:::::::::::'
             '::::::::::::'
                .::::::::::
           '::::::::::::::..
                ..::::::::::::.
             ``::::::::::::::::
               ::::``:::::::::'        .:::.              
              ::::'   ':::::'       .::::::::.
            .::::'      ::::     .:::::::'::::.
           .:::'       :::::  .:::::::::' ':::::.
          .::'        :::::.:::::::::'      ':::::.
         .::'         ::::::::::::::'         ``::::.
     ...:::           ::::::::::::'              ``::.
    ```` ':.          ':::::::::'                  ::::..
                       '.:::::'                    ':'````..
"""
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ZhengzhengLiu

import numpy as np

# x:数据集是一个numpy数组(每行代表一个数据点，每列代表一个特征值)，k:分类数，maxIt：迭代次数
def kmeans(X, k, maxIt):
    numPoints, numDim = X.shape  # 返回数据集X的行数和列数
    dataSet = np.zeros((numPoints, numDim + 1))  # 初始化新的数据集dataSet比X多一列，用来存放分标签
    dataSet[:, :-1] = X  # dataSet的所有行和除去最后一列的所有列的数值与X相同

    # 随机初始化k个中心点(利用randint函数从所有行实例中随机选出k个实例作为中心点)
    d_x=np.random.randint(numPoints, size=k)
    centroids = dataSet[d_x, :]
    # centroids = dataSet[0:2,:]       #初始化前两个实例作为中心点（也可以随机选取初始化中心点）

    # 为中心点最后一列初始化分类标记1到k
    centroids[:, -1] = range(1, k + 1)

    iterations = 0  # 循环次数
    oldCentroids = None  # 旧的中心点
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print("iterations:\n", iterations)  # 打印当前循环迭代次数
        print("dataSet:\n", dataSet)  # 打印当前数据集
        print("centroids:\n", centroids)  # 打印当前的中心

        oldCentroids = np.copy(centroids)  # 将当前中心点赋值到旧中心点中
        iterations += 1  # 迭代次数加1

        # 依照中心点为每个实例归类
        updateLabels(dataSet, centroids)

        # 根据归类后数据集和k值，计算新的中心点
        centroids = getCentroids(dataSet, k)

    return dataSet


# 迭代停止函数
def shouldStop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True  # 迭代次数比最大迭代次数大，则停止迭代
    return np.array_equal(oldCentroids, centroids)  # 比较新旧中心点的值是否相等，相等返回True，迭代终止


# 依照中心点为每个实例归类
def updateLabels(dataSet, centroids):
    numPoints, numDim = dataSet.shape  # 获取当前数据集的行列数
    for i in range(0, numPoints):
        # 当前行实例与中心点的距离最近的标记作为该实例的标记
        dataSet[i, -1] = getLabelFromClosesCentroid(dataSet[i, :-1], centroids)


def getLabelFromClosesCentroid(dataSetRow, centroids):
    label = centroids[0, -1]  # 初始化当前标记赋值为第一个中心点的标记（第一行最后一列）
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])  # 初始化计算当前行实例与第一个中心点实例的欧氏距离
    for i in range(1, centroids.shape[0]):  # 遍历第二个到最后一个中心点
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])  # 计算当前行实例与每一个中心点实例的欧氏距离
        if dist < minDist:
            minDist = dist
            label = centroids[i, -1]  # 若当前的欧氏距离比初始化的小，则取当前中心点的标记作为该实例标记
    print("minDist:", minDist)
    return label



##根据归类后数据集和k值，计算新的中心点
def getCentroids(dataSet, k):
    # 最后返回的新的中心点的值有k行，列数与dataSet相同
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k + 1):
        # 将所有标记是当前同一类的实例的数据组成一个类
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        result[i - 1, :-1] = np.mean(oneCluster, axis=0)  # 对同一类的实例求均值找出新的中心点
        result[i - 1, -1] = i

    return result

x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
testX = np.vstack((x1, x2, x3, x4))  # 垂直方向合并numpy数组

result = kmeans(testX, 2, 10)
print("final result:", result)
