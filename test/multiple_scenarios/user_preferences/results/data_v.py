# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter
#
#
# def prepare_data(df):
#     """确保数据为数值型并移除NaN值"""
#     df['Travel Time Cost (seconds)'] = pd.to_numeric(df['Travel Time Cost (seconds)'], errors='coerce')
#     df.dropna(subset=['Travel Time Cost (seconds)'], inplace=True)
#     return df
#
#
# # 加载数据
# aco_fixed_data = pd.read_csv('valid_aco_nopreference_results.csv')
# aco_random_data = pd.read_csv('valid_aco_noecar_results.csv')
# q_learning_fixed_data = pd.read_csv('valid_q_learning_noecar_results.csv')
# q_learning_random_data = pd.read_csv('valid_q_learning_noecar_results.csv')
#
# # 准备数据
# aco_fixed_data = prepare_data(aco_fixed_data)
# aco_random_data = prepare_data(aco_random_data)
# q_learning_fixed_data = prepare_data(q_learning_fixed_data)
# q_learning_random_data = prepare_data(q_learning_random_data)
#
# # 创建面积图
# fig, ax = plt.subplots(figsize=(10, 6))
# plt.rcParams.update({
#     'font.size': 18,      # Global font size
#     'axes.titlesize': 20, # Title font size
#     'axes.labelsize': 18, # X and Y axis labels font size
#     'xtick.labelsize': 18, # X tick labels font size
#     'ytick.labelsize': 18, # Y tick labels font size
#     'legend.fontsize': 18, # Legend font size
#     'figure.titlesize': 22 # Figure title font size
# })
#
# colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']  # 更鲜明的颜色方案
#
# # 使用正值绘制Q-Learning系列的数据
# ax.fill_between(range(len(q_learning_random_data)), 0, -q_learning_random_data['Travel Time Cost (seconds)'], step='mid', label='Q-Learning Excluded E-car Option', alpha=0.9, color=colors[3])
# ax.fill_between(range(len(q_learning_fixed_data)), 0, -q_learning_fixed_data['Travel Time Cost (seconds)'], step='mid', label='Q-Learning Default Preference Setting', alpha=0.7, color=colors[2])
#
# # 使用负值绘制ACO系列的数据
# ax.fill_between(range(len(aco_random_data)), 0, aco_random_data['Travel Time Cost (seconds)'], step='mid', label='ACO Excluded E-car Option', alpha=0.5, color=colors[1])
# ax.fill_between(range(len(aco_fixed_data)), 0, aco_fixed_data['Travel Time Cost (seconds)'], step='mid', label='ACO Default Preference Setting', alpha=1, color=colors[0])
#
# # 格式化y轴标签显示为正数
# formatter = FuncFormatter(lambda y, _: f'{abs(y):.0f}')
# ax.yaxis.set_major_formatter(formatter)
#
#
# ax.set_xlabel('OD pairs', fontsize=18)
# ax.set_ylabel('Travel Time Cost (seconds)', fontsize=18)
# ax.legend()
#
#
# plt.savefig('preferences.eps', format='eps')
# plt.show()

#
import pandas as pd
import matplotlib.pyplot as plt

def prepare_data(df):
    """确保数据为数值型并移除NaN值，并将Travel Time Cost放大100倍"""
    df['Travel Time Cost (seconds)'] = pd.to_numeric(df['Travel Time Cost (seconds)'], errors='coerce')
    df.dropna(subset=['Travel Time Cost (seconds)'], inplace=True)
    return df


# 加载数据
aco_fixed_data = pd.read_csv('final_valid_aco_nopreference_results.csv')
aco_random_data = pd.read_csv('ACO-noecar1.csv')
q_learning_fixed_data = pd.read_csv('final_valid_q_learning_nopreference_results.csv')
q_learning_random_data = pd.read_csv('final_valid_q_learning_noecar_results.csv')

# 准备数据
aco_fixed_data = prepare_data(aco_fixed_data)
aco_random_data = prepare_data(aco_random_data)
q_learning_fixed_data = prepare_data(q_learning_fixed_data)
q_learning_random_data = prepare_data(q_learning_random_data)

# 计算百分比
def calculate_percentage_less(data1, data2):
    return (data1 < data2).mean() * 100

# Q-Learning vs ACO in Random and Fixed scenarios
percentages = {
    'Q-Learning Excluded E-car Option < ACO Excluded E-car Option': calculate_percentage_less(q_learning_random_data['Travel Time Cost (seconds)'], aco_random_data['Travel Time Cost (seconds)']),
    'Q-Learning Default Preference Setting < ACO Default Preference Setting': calculate_percentage_less(q_learning_fixed_data['Travel Time Cost (seconds)'], aco_fixed_data['Travel Time Cost (seconds)']),
    'ACO Default Preference Setting < ACO Excluded E-car Option': calculate_percentage_less(aco_fixed_data['Travel Time Cost (seconds)'], aco_random_data['Travel Time Cost (seconds)']),
    'Q-Learning Default Preference Setting < Q-Learning Excluded E-car Option': calculate_percentage_less(q_learning_fixed_data['Travel Time Cost (seconds)'], q_learning_random_data['Travel Time Cost (seconds)']),
    'ACO Excluded E-car Option < Q-Learning Excluded E-car Option': calculate_percentage_less(aco_random_data['Travel Time Cost (seconds)'], q_learning_random_data['Travel Time Cost (seconds)']),
    'ACO Default Preference Setting < Q-Learning Default Preference Setting': calculate_percentage_less(aco_fixed_data['Travel Time Cost (seconds)'], q_learning_fixed_data['Travel Time Cost (seconds)']),

}

# 设置字体大小
plt.rcParams.update({'font.size': 14})
# 绘制柱状图
fig, ax = plt.subplots(figsize=(24, 6))
algorithms = list(percentages.keys())
values = list(percentages.values())
ax.barh(algorithms, values, color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948'])

# 设置图表标题和标签
# ax.set_title('Percentage of Travel Time Cost Less Than Other Algorithm')
ax.set_xlabel('Percentage (%)')
# ax.set_xlim(0, 100)  # 调整横轴范围以缩短横轴


# 显示每个条形上的百分比值
for i in range(len(values)):
    ax.text(values[i] + 1, i, f'{values[i]:.2f}%', va='center')

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)
# 调整布局以确保所有文本显示完全
plt.tight_layout()

# 保存图像并确保边界以文字为准
plt.savefig('percentage_comparison-1.png', bbox_inches='tight')

plt.show()
