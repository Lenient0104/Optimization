import pandas as pd
import matplotlib.pyplot as plt

# 加载CSV文件
df1 = pd.read_csv('../test/test_DQN/results/test-new.csv')
df2 = pd.read_csv('../test/test_q_learning/results/test.csv')

# 确保 "Find" 列为布尔型
df1['Find'] = df1['Find'].astype(bool)
df2['Find'] = df2['Find'].astype(bool)

# 合并数据集基于 'Experiment ID'
merged_df = pd.merge(df1[df1['Find']], df2[df2['Find']], on='Experiment ID', suffixes=('_file1', '_file2'))

# 根据 Experiment ID 排序
merged_df.sort_values('Experiment ID', inplace=True)

# 创建图形
plt.figure(figsize=(14, 7))

# 画折线图比较两个文件的Travel Time Cost
plt.plot(merged_df['Experiment ID'], merged_df['Travel Time Cost (seconds)_file1'], marker='o', linestyle='-', label='File 1')
plt.plot(merged_df['Experiment ID'], merged_df['Travel Time Cost (seconds)_file2'], marker='x', linestyle='--', label='File 2')

plt.xlabel('Experiment ID')
plt.ylabel('Travel Time Cost (seconds)')
plt.title('Line Chart Comparison of Travel Time Cost When Find is True')
plt.legend()
# plt.grid(True)
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout to not cut off labels
plt.show()