# 分类
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def score(c,g, path):

    data = np.loadtxt(path, dtype=float, delimiter=',')

    # 2.划分数据与标签
    x, y = np.split(data, indices_or_sections=(28,), axis=1)  # x为数据，y为标签
    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.7, test_size=0.3)

   # 降维
    pca = PCA(n_components=15)
    pca.fit(train_data)
    pca.fit(test_data)
    train_data = pca.transform(train_data)  # 训练数据集降维结果
    test_data = pca.transform(test_data)  # 训练数据集降维结果

    # 归一化
    ss = StandardScaler()
    train_data = ss.fit_transform(train_data)
    test_data = ss.fit_transform(test_data)


    # 3.训练svm分类器:linear,rbf,poly,sigmoid,precomputed
    classifier = svm.SVC(C=c, kernel='linear', gamma=g, decision_function_shape='ovr')

    # # ovr:一对多策略
    classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

    # 也可直接调用accuracy_score方法计算准确率
    from sklearn.metrics import accuracy_score

    tra_label = classifier.predict(train_data)  # 训练集的预测标签
    tes_label = classifier.predict(test_data)  # 测试集的预测标签
    P = accuracy_score(test_label, tes_label)
    # print("训练集：", accuracy_score(train_label, tra_label))
    # print("测试集：", accuracy_score(test_label, tes_label))
    print("测试", P)
    return P

if __name__ == '__main__':

    path = r'E:/benkebiye/test.csv'
    P = score(1,1,path)
    print("P:", P)



