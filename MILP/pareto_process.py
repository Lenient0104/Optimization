import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('macOSX')  # 或者 'Qt5Agg', 'macOSX' 等，根据您的系统环境选择一个合适的后端
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('pareto_values1004.csv')

# Dropping duplicates
df_unique = df.drop_duplicates()

# Plotting scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_unique["Time cost"], df_unique["Fees"])
plt.xlabel('Time cost')
plt.ylabel('Fees')
plt.title('Scatter Plot of Time Cost vs Fees')
plt.grid(True)
plt.show()