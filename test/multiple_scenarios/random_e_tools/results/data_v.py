# import pandas as pd
# import matplotlib.pyplot as plt
#
# def prepare_data(df):
#     """确保数据为数值型并移除NaN值"""
#     df['Travel Time Cost (seconds)'] = pd.to_numeric(df['Travel Time Cost (seconds)'], errors='coerce')
#     df.dropna(subset=['Travel Time Cost (seconds)'], inplace=True)
#     return df
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
# fig, axs = plt.subplots(2, 1, figsize=(20, 16))  # 创建两个子图
#
# colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']  # 更鲜明的颜色方案
#
# # 第一个子图用于Q-Learning数据
# axs[0].fill_between(range(len(q_learning_random_data)), 0, q_learning_random_data['Travel Time Cost (seconds)'], step='mid', label='Q-Learning Random', alpha=0.9, color=colors[3])
# axs[0].fill_between(range(len(q_learning_fixed_data)), 0, q_learning_fixed_data['Travel Time Cost (seconds)'], step='mid', label='Q-Learning Fixed', alpha=0.7, color=colors[2])
# axs[0].set_title('Q-Learning Travel Time Cost')
# axs[0].set_xlabel('OD pairs', fontsize=14)
# axs[0].set_ylabel('Travel Time Cost (seconds)', fontsize=14)
# axs[0].legend()
#
# # 第二个子图用于ACO数据
# axs[1].fill_between(range(len(aco_random_data)), 0, aco_random_data['Travel Time Cost (seconds)'], step='mid', label='ACO Random', alpha=0.5, color=colors[1])
# axs[1].fill_between(range(len(aco_fixed_data)), 0, aco_fixed_data['Travel Time Cost (seconds)'], step='mid', label='ACO Fixed', alpha=1, color=colors[0])
# axs[1].set_title('ACO Travel Time Cost')
# axs[1].set_xlabel('OD pairs', fontsize=14)
# axs[1].set_ylabel('Travel Time Cost (seconds)', fontsize=14)
# axs[1].legend()
#
# plt.tight_layout()  # 调整布局，避免子图间内容重叠
# plt.savefig('e-tools distribution.png')
# plt.show()


# import pandas as pd
import matplotlib.pyplot as plt

# def prepare_data(df):
#     """确保数据为数值型并移除NaN值，并将Travel Time Cost放大100倍"""
#     df['Travel Time Cost (seconds)'] = pd.to_numeric(df['Travel Time Cost (seconds)'], errors='coerce') * 100
#     df.dropna(subset=['Travel Time Cost (seconds)'], inplace=True)
#     return df
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
# # 计算百分比
# def calculate_percentage_less(data1, data2):
#     return (data1 < data2).mean() * 100
#
# # Q-Learning vs ACO in Random and Fixed scenarios
# percentages = {
#     'Q-Learning Random < ACO Random': calculate_percentage_less(q_learning_random_data['Travel Time Cost (seconds)'], aco_random_data['Travel Time Cost (seconds)']),
#     'Q-Learning Fixed < ACO Fixed': calculate_percentage_less(q_learning_fixed_data['Travel Time Cost (seconds)'], aco_fixed_data['Travel Time Cost (seconds)']),
#     'ACO Fixed < ACO Random': calculate_percentage_less(aco_fixed_data['Travel Time Cost (seconds)'],
#                                                         aco_random_data['Travel Time Cost (seconds)']),
#     'Q-Learning Fixed < Q-Learning Random': calculate_percentage_less(
#         q_learning_fixed_data['Travel Time Cost (seconds)'], q_learning_random_data['Travel Time Cost (seconds)']),
#     'ACO Random < Q-Learning Random': calculate_percentage_less(aco_random_data['Travel Time Cost (seconds)'], q_learning_random_data['Travel Time Cost (seconds)']),
#     'ACO Fixed < Q-Learning Fixed': calculate_percentage_less(aco_fixed_data['Travel Time Cost (seconds)'], q_learning_fixed_data['Travel Time Cost (seconds)']),
#
# }
#
# # 设置字体大小
# plt.rcParams.update({'font.size': 14})
# # 绘制柱状图
# fig, ax = plt.subplots(figsize=(24, 6))
# algorithms = list(percentages.keys())
# values = list(percentages.values())
# ax.barh(algorithms, values, color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948'])
#
# # 设置图表标题和标签
# # ax.set_title('Percentage of Travel Time Cost Less Than Other Algorithm')
# ax.set_xlabel('Percentage (%)')
# # ax.set_xlim(0, 100)  # 调整横轴范围以缩短横轴
#
#
# # 显示每个条形上的百分比值
# for i in range(len(values)):
#     ax.text(values[i] + 1, i, f'{values[i]:.2f}%', va='center')
#
# for spine in ax.spines.values():
#     spine.set_edgecolor('black')
#     spine.set_linewidth(1.5)
# # 调整布局以确保所有文本显示完全
# plt.tight_layout()
#
# # 保存图像并确保边界以文字为准
# plt.savefig('percentage_comparison.png', bbox_inches='tight')
#
# plt.show()
#
# import pandas as pd
# import matplotlib.pyplot as plt
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
# # 计算Random和Fixed的差值
# aco_difference = aco_random_data['Travel Time Cost (seconds)'] - aco_fixed_data['Travel Time Cost (seconds)']
# q_learning_difference = q_learning_random_data['Travel Time Cost (seconds)'] - q_learning_fixed_data['Travel Time Cost (seconds)']
#
# # 创建线图
# fig, ax = plt.subplots(figsize=(20, 8))
#
# colors = ['#4E79A7', '#E15759']  # 分配不同的颜色
#
# ax.plot(range(len(aco_difference)), aco_difference, label='ACO Random-Fixed Difference', color=colors[0])
# ax.plot(range(len(q_learning_difference)), q_learning_difference, label='Q-Learning Random-Fixed Difference', color=colors[1])
# # 在y=0处添加一条横线
# ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
#
# ax.set_title('Random vs Fixed Configuration Differences')
# ax.set_xlabel('OD pairs', fontsize=14)
# ax.set_ylabel('Difference in Travel Time Cost (seconds)', fontsize=14)
# ax.legend()
#
# plt.savefig('random_vs_fixed_configuration.png')
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
#
# def prepare_data(df):
#     """确保数据为数值型并移除NaN值"""
#     df['Travel Time Cost (seconds)'] = pd.to_numeric(df['Travel Time Cost (seconds)'], errors='coerce')
#     df.dropna(subset=['Travel Time Cost (seconds)'], inplace=True)
#     return df
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
# # 计算Random和Fixed的差值
# aco_difference = aco_random_data['Travel Time Cost (seconds)'] - aco_fixed_data['Travel Time Cost (seconds)']
# q_learning_difference = q_learning_random_data['Travel Time Cost (seconds)'] - q_learning_fixed_data['Travel Time Cost (seconds)']
#
# # 创建线图
# fig, ax = plt.subplots(figsize=(20, 8))
#
# colors = ['#4E79A7', '#E15759']  # 分配不同的颜色
#
# # 在同一个图中画两条线，调整每条线的起点为其最小值
# aco_min = aco_difference.min()
# q_learning_min = q_learning_difference.min()
#
# ax.plot(range(len(aco_difference)), aco_difference - aco_min, label='ACO Random-Fixed Difference', color=colors[0])
# ax.plot(range(len(q_learning_difference)), q_learning_difference - q_learning_min, label='Q-Learning Random-Fixed Difference', color=colors[1])
#
# ax.set_title('Random vs Fixed Configuration Differences')
# ax.set_xlabel('OD pairs', fontsize=14)
# ax.set_ylabel('Difference in Travel Time Cost (seconds)', fontsize=14)
# ax.legend()
#
# plt.savefig('random_vs_fixed_configuration-1.png')
# plt.show()
#
#
#
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
fig, ax = plt.subplots(figsize=(10, 6))
plt.rcParams.update({
    'font.size': 18,      # Global font size
    'axes.titlesize': 20, # Title font size
    'axes.labelsize': 18, # X and Y axis labels font size
    'xtick.labelsize': 18, # X tick labels font size
    'ytick.labelsize': 18, # Y tick labels font size
    'legend.fontsize': 18, # Legend font size
    'figure.titlesize': 22 # Figure title font size
})

colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']  # 更鲜明的颜色方案

# 使用正值绘制Q-Learning系列的数据
ax.fill_between(range(len(q_learning_random_data)), 0, -q_learning_random_data['Travel Time Cost (seconds)'], step='mid', label='Q-Learning Random', alpha=0.9, color=colors[3])
ax.fill_between(range(len(q_learning_fixed_data)), 0, -q_learning_fixed_data['Travel Time Cost (seconds)'], step='mid', label='Q-Learning Fixed', alpha=0.7, color=colors[2])

# 使用负值绘制ACO系列的数据
ax.fill_between(range(len(aco_random_data)), 0, aco_random_data['Travel Time Cost (seconds)'], step='mid', label='ACO Random', alpha=0.5, color=colors[1])
ax.fill_between(range(len(aco_fixed_data)), 0, aco_fixed_data['Travel Time Cost (seconds)'], step='mid', label='ACO Fixed', alpha=1, color=colors[0])

# 格式化y轴标签显示为正数
formatter = FuncFormatter(lambda y, _: f'{abs(y):.0f}')
ax.yaxis.set_major_formatter(formatter)


ax.set_xlabel('OD pairs', fontsize=18)
ax.set_ylabel('Travel Time Cost (seconds)', fontsize=18)
ax.legend()

plt.savefig('etools.eps', format='eps')
plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
#
# def prepare_data(df):
#     """确保数据为数值型并移除NaN值，并将Travel Time Cost放大100倍"""
#     df['Travel Time Cost (seconds)'] = pd.to_numeric(df['Travel Time Cost (seconds)'], errors='coerce') * 100
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
# # 计算百分比
# def calculate_percentage_less(data1, data2):
#     return (data1 < data2).mean() * 100
#
# # Q-Learning vs ACO in Random and Fixed scenarios
# percentages = {
#     'Q-Learning Random < ACO Random': calculate_percentage_less(q_learning_random_data['Travel Time Cost (seconds)'], aco_random_data['Travel Time Cost (seconds)']),
#     'Q-Learning Fixed < ACO Fixed': calculate_percentage_less(q_learning_fixed_data['Travel Time Cost (seconds)'], aco_fixed_data['Travel Time Cost (seconds)']),
#     'ACO Fixed < ACO Random': calculate_percentage_less(aco_fixed_data['Travel Time Cost (seconds)'],
#                                                         aco_random_data['Travel Time Cost (seconds)']),
#     'Q-Learning Fixed < Q-Learning Random': calculate_percentage_less(
#         q_learning_fixed_data['Travel Time Cost (seconds)'], q_learning_random_data['Travel Time Cost (seconds)']),
#     'ACO Random < Q-Learning Random': calculate_percentage_less(aco_random_data['Travel Time Cost (seconds)'], q_learning_random_data['Travel Time Cost (seconds)']),
#     'ACO Fixed < Q-Learning Fixed': calculate_percentage_less(aco_fixed_data['Travel Time Cost (seconds)'], q_learning_fixed_data['Travel Time Cost (seconds)']),
#
# }
#
# # 设置字体大小
# plt.rcParams.update({'font.size': 14})
# # 绘制柱状图
# fig, ax = plt.subplots(figsize=(10, 6))
# algorithms = list(percentages.keys())
# values = list(percentages.values())
# ax.barh(algorithms, values, color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948'])
#
# # 设置图表标题和标签
# # ax.set_title('Percentage of Travel Time Cost Less Than Other Algorithm')
# ax.set_xlabel('Percentage (%)')
# # ax.set_xlim(0, 100)  # 调整横轴范围以缩短横轴
#
#
# # 显示每个条形上的百分比值
# for i in range(len(values)):
#     ax.text(values[i] + 1, i, f'{values[i]:.2f}%', va='center')
#
# for spine in ax.spines.values():
#     spine.set_edgecolor('black')
#     spine.set_linewidth(1.5)
# # 调整布局以确保所有文本显示完全
# plt.tight_layout()
#
# # 保存图像并确保边界以文字为准
# plt.savefig('percentage_comparison.png', bbox_inches='tight')
#
# plt.show()
