# 分类
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np



def score(c,g, path):

    data = np.loadtxt(path, dtype=float, delimiter=',')
    # 降维
    pca = PCA(n_components=5)
    pca.fit(data)
    data = pca.transform(data)  # 训练数据集降维结果
    np.savetxt('E:/benkebiye/PCA.csv', data, delimiter=',',fmt='%.5f')
    print('After pca: \n', data[:3, :], '\n')

    # # 归一化
    # ss = StandardScaler()
    # data = ss.fit_transform(data)
    P=1

    return P

if __name__ == '__main__':

    path = r'E:/benkebiye/test.csv'
    P = score(11,0.5,path)
    print("P:", P)



