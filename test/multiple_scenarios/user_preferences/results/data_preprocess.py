import pandas as pd

# 加载数据
aco_data = pd.read_csv('ACO-noecar-0611.csv')
# q_learning_data = pd.read_csv('Updated_Q_1.csv')

# 处理'Find'列的NA/NAN值，将其视为False
aco_data['Find'].fillna(False, inplace=True)
# q_learning_data['Find'].fillna(False, inplace=True)

# 分离Fixed和Random数据，并重新编号
aco_fixed = aco_data[aco_data['User Preference'] == 'No Preference'].reset_index(drop=True)
aco_random = aco_data[aco_data['User Preference'] == 'Prefer No E-car Option'].reset_index(drop=True)
q_learning_fixed = q_learning_data[q_learning_data['User Preference'] == 'No Preference'].reset_index(drop=True)
q_learning_random = q_learning_data[q_learning_data['User Preference'] == 'Prefer No E-car Option'].reset_index(drop=True)

# 添加实验编号
aco_fixed['experiment_index'] = aco_fixed.index + 1
aco_random['experiment_index'] = aco_random.index + 1
q_learning_fixed['experiment_index'] = q_learning_fixed.index + 1
q_learning_random['experiment_index'] = q_learning_random.index + 1

# 筛选出同时有效的实验编号
valid_fixed = set(aco_fixed[aco_fixed['Find']]['experiment_index']).intersection(set(q_learning_fixed[q_learning_fixed['Find']]['experiment_index']))
valid_random = set(aco_random[aco_random['Find']]['experiment_index']).intersection(set(q_learning_random[q_learning_random['Find']]['experiment_index']))

# 交叉验证所有四种情况，确保实验编号在所有情况下都有效
valid_indices = valid_fixed.intersection(valid_random)

# 保留有效的实验数据
aco_fixed_valid = aco_fixed[aco_fixed['experiment_index'].isin(valid_indices)]
aco_random_valid = aco_random[aco_random['experiment_index'].isin(valid_indices)]
q_learning_fixed_valid = q_learning_fixed[q_learning_fixed['experiment_index'].isin(valid_indices)]
q_learning_random_valid = q_learning_random[q_learning_random['experiment_index'].isin(valid_indices)]

# 输出每种情况的数据行数
print("ACO Fixed Valid Experiments Count:", aco_fixed_valid.shape[0])
print("ACO Random Valid Experiments Count:", aco_random_valid.shape[0])
print("Q-learning Fixed Valid Experiments Count:", q_learning_fixed_valid.shape[0])
print("Q-learning Random Valid Experiments Count:", q_learning_random_valid.shape[0])

# 保存结果
aco_fixed_valid.to_csv('valid_aco_nopreference_results.csv', index=False)
aco_random_valid.to_csv('valid_aco_noecar_results.csv', index=False)
q_learning_fixed_valid.to_csv('valid_q_learning_noecar_results.csv', index=False)
q_learning_random_valid.to_csv('valid_q_learning_noecar_results.csv', index=False)

# 输出统计
print("Total Valid Experiments Count:", len(valid_indices))
