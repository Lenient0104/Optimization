import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt


# 已知的极端点
cost_min = 0  # 成本最小化极端点是0
time_at_cost_min = 5405  # 假设成本最小时的时间
cost_at_time_min = 265  # 假设时间最小时的成本
time_min = 3354.7711735061193  # 时间最小化极端点

# 生成数据点
times = np.linspace(time_min, time_at_cost_min, 100)
costs = np.interp(times, [time_min, time_at_cost_min], [cost_at_time_min, cost_min])

# 多项式拟合
coefficients = np.polyfit(times, costs, 2)  # 二次多项式拟合
polynomial = np.poly1d(coefficients)
costs_fitted = polynomial(times)

# 绘制帕累托前沿
plt.figure(figsize=(10, 6))
plt.scatter(times, costs, color='blue', label='Interpolated Points')
plt.plot(times, costs_fitted, color='red', label='Fitted Polynomial')
plt.title('Pareto Front: Cost vs Time')
plt.xlabel('Time')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
plt.show()