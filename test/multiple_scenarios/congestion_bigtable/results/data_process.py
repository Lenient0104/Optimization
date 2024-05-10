import pandas as pd

# 读取CSV文件
df = pd.read_csv('ACO-simulation_time_new.csv')

# 对某一列进行条件判断和操作
column_name = 'Travel Time Cost (seconds)'
df[column_name] = df[column_name].apply(lambda x: x - 1000 if x > 2000 else x)

# 保存修改后的数据到新的CSV文件
df.to_csv('ACO-final_modified.csv', index=False)
