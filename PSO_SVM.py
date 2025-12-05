# 粒子群优化颜色指数
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", family='Microsoft YaHei')
from machine_learn.svm_cls import score
import  time

def fitness_func(X):
    c = X[:, 0]
    g = X[:, 1]
    path = r'E:/benkebiye/test.csv'
    list = []
    for i in range(c.size):
        P = score(c[i],g[i],path)
        list.append(1-P)

    return np.array(list)


def velocity_update(V, X, pbest, gbest, c1, c2, w, max_val):
    size = X.shape[0]  # 返回矩阵X的行数
    r1 = np.random.random((size, 1))  # 该函数表示成size行 1列的浮点数，浮点数都是从0-1中随机。
    r2 = np.random.random((size, 1))
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)  # 注意这里得到的是一个矩阵
    # 这里是一个防止速度过大的处理，怕错过最理想值
    V[V < -max_val] = -max_val
    V[V > max_val] = max_val
    return V

# 更新粒子位置，根据公式X(t+1)=X(t)+V
def position_updata(X, V):
    return X + V

def pos():
    w = 1  # 设置惯性权重
    c1 = 0.8  # 设置个体学习系数
    c2 = 0.5  # 设置全局学习系数
    r1 = 2
    r2 = 2
    dim = 2    #维度
    size = 10  # 初始化粒子群数
    iter_num = 1000  # 迭代次数
    max_val = 0.01  # 限定最大速度为0.5
    #best_fitness = float(9e10)  # 初始化适应度的值
    fitness_val_list = []

    # 初始化各个粒子的位置
    X = np.random.uniform(0, 50, size=(size, dim))
    # 初始化各个粒子的速度
    V = np.random.uniform(-0.05, 0.05, size=(size, dim))

    p_fitness = fitness_func(X)  # 得到各个个体的适应度值
    g_fitness = p_fitness.min()  # 全局最理想的适应度值
    fitness_val_list.append(g_fitness)

    pbest = X  # 初始化个体的最优位置
    gbest = X[p_fitness.argmin()]  # 初始化整个整体的最优位置

    # 迭代
    for i in range(1, iter_num):
        time_start = time.time()  # 开始计时
        print("当前迭代次数：", i)
        V = velocity_update(V, X, pbest, gbest, c1, c2, w, max_val)
        X = position_updata(X, V)
        p_fitness2 = fitness_func(X)
        g_fitness2 = p_fitness2.min()
        # 更新每个粒子的历史的最优位置
        for j in range(size):
            if p_fitness[j] > p_fitness2[j]:
                pbest[j] = X[j]
                p_fitness[j] = p_fitness2[j]
            if g_fitness > g_fitness2:
                gbest = X[p_fitness2.argmin()]
                g_fitness = g_fitness2
            fitness_val_list.append(g_fitness)
        time_end = time.time()  # 结束计时
        time_c = time_end - time_start  # 运行所花时间
        print('time cost', time_c, 's')



    print("最优值是：%.5f" % (fitness_val_list[-1]))
    print("最优解是：x=%.5f,y=%.5f" % (gbest[0],gbest[1]))

    plt.plot(fitness_val_list, c='r')
    plt.title('迭代过程')
    plt.savefig('E:/DATA/JY/ML_DATA/GY/plot.png', dpi=300)

    plt.show()

if __name__ == '__main__':

    pos()