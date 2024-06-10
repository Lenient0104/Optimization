import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("Q_learning_results_parameters.csv")

# 选择要减去的列，例如 "Time Cost"
column_to_subtract = "Time Cost"

# 减去的数值
subtract_value = 2000

# 将所选列中的每个值减去给定的值
df[column_to_subtract] = df[column_to_subtract] - subtract_value

# 显示修改后的 DataFrame
print(df)

# 如果需要，将修改后的 DataFrame 保存回 CSV 文件
df.to_csv("modified_Q-parameter-file.csv", index=False)
