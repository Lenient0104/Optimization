import pandas as pd
import matplotlib.pyplot as plt

# 读取已经过滤的数据
file_path = 'filtered_data.csv'
filtered_data = pd.read_csv(file_path)

# 提取0.07、0.1和1.0能量的数据
energy_0_07 = filtered_data[filtered_data['Initial Energy'] == 0.07]
energy_0_1 = filtered_data[filtered_data['Initial Energy'] == 0.10]
energy_1_0 = filtered_data[filtered_data['Initial Energy'] == 1.00]

# 合并0.07和0.1能量的数据，确保Experiment ID匹配
merged_data_0_07_0_1 = pd.merge(energy_0_07, energy_0_1, on='Experiment ID', suffixes=('_0.07', '_0.1'))
# 合并0.1和1.0能量的数据，确保Experiment ID匹配
merged_data_0_1_1_0 = pd.merge(energy_0_1, energy_1_0, on='Experiment ID', suffixes=('_0.1', '_1.0'))

# 计算Travel Time Cost的差值
merged_data_0_07_0_1['Diff_0.07-0.1'] = merged_data_0_07_0_1['Travel Time Cost (seconds)_0.07'] - merged_data_0_07_0_1['Travel Time Cost (seconds)_0.1']
merged_data_0_1_1_0['Diff_0.1-1.0'] = merged_data_0_1_1_0['Travel Time Cost (seconds)_0.1'] - merged_data_0_1_1_0['Travel Time Cost (seconds)_1.0']

# 计算正值和负值的数量
diff_0_07_0_1_positive = (merged_data_0_07_0_1['Diff_0.07-0.1'] > 0).sum()
diff_0_07_0_1_negative = (merged_data_0_07_0_1['Diff_0.07-0.1'] < 0).sum()

diff_0_1_1_0_positive = (merged_data_0_1_1_0['Diff_0.1-1.0'] > 0).sum()
diff_0_1_1_0_negative = (merged_data_0_1_1_0['Diff_0.1-1.0'] < 0).sum()

# 输出结果
print(f"0.07 - 0.1 Energy: Positive values = {diff_0_07_0_1_positive}, Negative values = {diff_0_07_0_1_negative}")
print(f"0.1 - 1.0 Energy: Positive values = {diff_0_1_1_0_positive}, Negative values = {diff_0_1_1_0_negative}")
