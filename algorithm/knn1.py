# 从sklearn中导入neighbors模块
from sklearn import neighbors
# 导入已经存在的数据集
from sklearn import datasets

# 调用KNN分类器
knn = neighbors.KNeighborsClassifier()

# 得到iris数据库
iris = datasets.load_iris()

print (iris)

# 第一个参数为特征值,第二个参数为前面每一行对应的分类结果;建立模型
knn.fit(iris.data, iris.target)

# 通过建立好的模型,对新的花瓣类别进行预测
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])

print (predictedLabel)