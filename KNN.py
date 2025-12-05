from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# 1. 读取数据集
path = r'E:/benkebiye/test.csv'
data = np.loadtxt(path, dtype=float, delimiter=',')

# 2. 划分数据与标签
x, y = np.split(data, indices_or_sections=(28,), axis=1)  # x为数据，y为标签
train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.7, test_size=0.3)

# 3. 归一化
ss = StandardScaler()
train_data = ss.fit_transform(train_data)
test_data = ss.transform(test_data)

# 4. 创建KNN分类器
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # 设置邻居数量为5

# 5. 训练模型
knn_classifier.fit(train_data, train_label.ravel())

# 6. 在测试集上进行预测
test_pred = knn_classifier.predict(test_data)

# 7. 计算准确率
accuracy = accuracy_score(test_label, test_pred)
print("Accuracy:", accuracy)
