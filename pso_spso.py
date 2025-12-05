import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'无法显示的问题

def objective_function(x, y):
    return x**2 + y**2+x

def decreasing_sigmoid(x, k, mid):
    return 2/ (1 + np.exp(k * (x - mid)))

dim = 2
max_iter = 35
num_particles = 30
particles = np.random.uniform(-10, 10, size=(num_particles, dim))
velocities = np.random.uniform(-1, 1, size=(num_particles, dim))
def standard_pso_algorithm(particles,velocities,max_iter, num_particles, dim):
    # 初始化粒子群
    pbest_positions = particles.copy()
    pbest_values = np.array([objective_function(x, y) for x, y in particles])
    gbest_index = np.argmin(pbest_values)
    gbest_value = pbest_values[gbest_index]
    gbest_position = pbest_positions[gbest_index]
    fitness_values = [gbest_value]
    w = 0.9  # 惯性权重
    c1, c2 = 1.5, 1.5  # 学习因子
    # 迭代
    for iteration in range(max_iter):
        fitness_values.append(gbest_value)
        r1, r2 = np.random.random(size=(num_particles, dim)), np.random.random(size=(num_particles, dim))
        velocities = w * velocities + c1 * r1 * (pbest_positions - particles) + c2 * r2 * (gbest_position - particles)
        particles = particles + velocities
        current_values = np.array([objective_function(x, y) for x, y in particles])
        # 更新个体最优位置和全局最优位置
        update_indices = current_values < pbest_values
        pbest_positions[update_indices] = particles[update_indices]
        pbest_values[update_indices] = current_values[update_indices]
        if np.min(current_values) < gbest_value:
            gbest_index = np.argmin(current_values)
            gbest_value = current_values[gbest_index]
            gbest_position = particles[gbest_index]

    return fitness_values

def custom_pso_algorithm(particles,velocities,max_iter, num_particles, dim, w_max, w_min, k, mid):
    # 初始化粒子群

    pbest_positions = particles.copy()
    pbest_values = np.array([objective_function(x, y) for x, y in particles])
    gbest_index = np.argmin(pbest_values)
    gbest_value = pbest_values[gbest_index]
    gbest_position = pbest_positions[gbest_index]
    fitness_values = [gbest_value]
    # 迭代
    for iteration in range(max_iter):
        fitness_values.append(gbest_value)
        current_w = decreasing_sigmoid(iteration, k, mid) * (w_max - w_min) + w_min
        r1, r2 = np.random.random(size=(num_particles, dim)), np.random.random(size=(num_particles, dim))
        velocities = current_w * velocities + r1 * (pbest_positions - particles) + r2 * (gbest_position - particles)
        particles = particles + velocities

        current_values = np.array([objective_function(x, y) for x, y in particles])

        # 更新个体最优位置和全局最优位置
        update_indices = current_values < pbest_values
        pbest_positions[update_indices] = particles[update_indices]
        pbest_values[update_indices] = current_values[update_indices]

        if np.min(current_values) < gbest_value:
            gbest_index = np.argmin(current_values)
            gbest_value = current_values[gbest_index]
            gbest_position = particles[gbest_index]
    return fitness_values

# 设置参数
w_max, w_min, k, mid =1.6, 0.1, 1, max_iter /2

# 运行标准PSO算法和非线性动态惯性权重PSO算法
standard_pso_values = standard_pso_algorithm(particles,velocities,max_iter, num_particles, dim)
custom_pso_values = custom_pso_algorithm(particles,velocities,max_iter, num_particles, dim, w_max, w_min, k, mid)

# 绘制适应度值的变化
plt.figure(figsize=(8, 6))
plt.plot(standard_pso_values, color='b', marker='o', linestyle='-', linewidth=2, markersize=6, label='标准PSO')
plt.plot(custom_pso_values, color='r', marker='o', linestyle='-', linewidth=2, markersize=6, label='非线性动态PSO')
plt.xlabel('迭代次数', fontsize=14)
plt.ylabel('最优适应度值', fontsize=14)
plt.title('标准PSO与非线性动态PSO优化比较', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

print("标准PSO最优值 f(x, y):", standard_pso_values[-1])
print("非线性动态PSO最优值 f(x, y):", custom_pso_values[-1])
