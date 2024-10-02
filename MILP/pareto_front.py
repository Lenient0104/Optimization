import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# 目标函数定义
def cost(x1, x2):
    return (x1 - 2)**2 + (x2 - 3)**2

def quality(x1, x2):
    return -((x1 - 4)**2 + (x2 - 1)**2)

# 生成网格点
x1_values = np.linspace(0, 5, 100)
x2_values = np.linspace(0, 5, 100)
X1, X2 = np.meshgrid(x1_values, x2_values)

# 计算目标函数值
F1 = cost(X1, X2)
F2 = quality(X1, X2)

# 创建列表存储Pareto最优解
pareto_solutions = []

# 网格搜索找到近似的Pareto前沿
for i in range(100):
    for j in range(100):
        dominated = False
        for k in range(100):
            for l in range(100):
                if (F1[k, l] <= F1[i, j] and F2[k, l] >= F2[i, j]) and (F1[k, l] < F1[i, j] or F2[k, l] > F2[i, j]):
                    dominated = True
                    break
            if dominated:
                break
        if not dominated:
            pareto_solutions.append((X1[i, j], X2[i, j], F1[i, j], F2[i, j]))

# 提取Pareto前沿的目标值
pareto_cost = [solution[2] for solution in pareto_solutions]
pareto_quality = [solution[3] for solution in pareto_solutions]


# 绘制Pareto前沿
plt.figure(figsize=(10, 6))
plt.scatter(pareto_cost, pareto_quality, color='red', label='Pareto Front')
plt.xlabel('Cost')
plt.ylabel('Quality')
plt.title('Pareto Frontier for Cost vs Quality')
plt.legend()
plt.grid()
plt.savefig('pareto.png')
plt.show()

