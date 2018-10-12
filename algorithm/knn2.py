# coding=utf-8
import csv
import random
import operator
import numpy as np

# 加载数据集,将原始数据集分为训练集和测试集
# filename数据集所在的文件名
# split根据此数值将数据集分为测试集和训练集
# trainingSet训练集,testSet测试集
def loadDataset(filename, split, trainingSet = [], testSet = []):
    # 读取文件为csvfile
    with open(filename, 'r') as csvfile:
        # 把读进来的文件转为行的格式
        lines = csv.reader(csvfile)
        # 把读进来的所有行转换成list的格式
        dataset = list(lines)
        # 将数据集分为训练集与测试集
        for x in range(len(dataset)-1):
            # y :0,1,2,3
            for y in range(4):
                # 将加载数据由string转为double
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

# 计算距离
# instance12分别为两个实例
# length为要计算的维度
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]), 2)
    return np.sqrt(distance)

# 返回最近的K个label,从训练集中选出k个离测试实例最近的实例
# trainingSet训练集
# testInstance测试实例
# k选出k个
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        #testinstance
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
        #distances.append(dist)
    # 从小到大排序
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        return neighbors

# 根据邻居得到邻居所属类别最多分类
def getResponse(neighbors):
    # print neighbors
    classVotes = {}
    for x in range(len(neighbors)):
        # -1代表取最后一个值
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# 预测分类后正确率
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0


def main():
    # 准备数据和数据预处理
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset(r'irisdata.txt', split, trainingSet, testSet)
    print ('Train set: ' + repr(len(trainingSet)))
    print ('Test set: ' + repr(len(testSet)))
    # 预测结果
    predictions = []
    k = 3
    for x in range(len(testSet)):
        # trainingsettrainingSet[x]
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print ('>predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    print ('predictions: ' + repr(predictions))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()