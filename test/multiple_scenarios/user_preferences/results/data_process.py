import pandas as pd

# 读取CSV文件
df = pd.read_csv('input.csv')

# 确保新列与现有数据行数对齐
num_rows_to_add = len(df)
new_column_data = ['value1', 'value2', 'value3'] + [pd.NA] * (num_rows_to_add - len(new_column_data))

# 添加新列
df['New Column'] = new_column_data

# 将DataFrame写回CSV文件
df.to_csv('output.csv', index=False)
