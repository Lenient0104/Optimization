import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('macOSX')  # 或者 'Qt5Agg', 'macOSX' 等，根据您的系统环境选择一个合适的后端
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('pareto_values1006.csv')

# Dropping duplicates
df_unique = df.drop_duplicates()

# 排序（可选，如果想让线条按照时间或费用递增）
df_unique = df_unique.sort_values(by="Time cost")

# Plotting line plot
plt.figure(figsize=(10, 6))
plt.plot(df_unique["Time cost"], df_unique["Fees"], marker='o')  # 添加 marker='o' 以显示每个点的位置
plt.xlabel('Time cost')
plt.ylabel('Fees')
plt.title('Line Plot of Time Cost vs Fees')
plt.grid(True)
plt.show()
