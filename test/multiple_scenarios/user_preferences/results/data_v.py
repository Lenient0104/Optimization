import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def prepare_data(df):
    """确保数据为数值型并移除NaN值"""
    df['Travel Time Cost (seconds)'] = pd.to_numeric(df['Travel Time Cost (seconds)'], errors='coerce')
    df.dropna(subset=['Travel Time Cost (seconds)'], inplace=True)
    return df


# 加载数据
aco_fixed_data = pd.read_csv('valid_aco_fixed_results.csv')
aco_random_data = pd.read_csv('valid_aco_random_results.csv')
q_learning_fixed_data = pd.read_csv('valid_q_learning_fixed_results.csv')
q_learning_random_data = pd.read_csv('valid_q_learning_random_results.csv')

# 准备数据
aco_fixed_data = prepare_data(aco_fixed_data)
aco_random_data = prepare_data(aco_random_data)
q_learning_fixed_data = prepare_data(q_learning_fixed_data)
q_learning_random_data = prepare_data(q_learning_random_data)

# 创建面积图
fig, ax = plt.subplots(figsize=(20, 8))

colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']  # 更鲜明的颜色方案

# 使用正值绘制Q-Learning系列的数据
ax.fill_between(range(len(q_learning_random_data)), 0, -q_learning_random_data['Travel Time Cost (seconds)'], step='mid', label='Q-Learning Excluded E-car Option', alpha=0.9, color=colors[3])
ax.fill_between(range(len(q_learning_fixed_data)), 0, -q_learning_fixed_data['Travel Time Cost (seconds)'], step='mid', label='Q-Learning Default Preference Setting', alpha=0.7, color=colors[2])

# 使用负值绘制ACO系列的数据
ax.fill_between(range(len(aco_random_data)), 0, aco_random_data['Travel Time Cost (seconds)'], step='mid', label='ACO Excluded E-car Option', alpha=0.5, color=colors[1])
ax.fill_between(range(len(aco_fixed_data)), 0, aco_fixed_data['Travel Time Cost (seconds)'], step='mid', label='ACO Default Preference Setting', alpha=1, color=colors[0])

# 格式化y轴标签显示为正数
formatter = FuncFormatter(lambda y, _: f'{abs(y):.0f}')
ax.yaxis.set_major_formatter(formatter)


ax.set_xlabel('OD pairs', fontsize=14)
ax.set_ylabel('Travel Time Cost (seconds)', fontsize=14)
ax.legend()
plt.savefig('preferences.png')
plt.show()

