# Identifying valid experiments where 'Find' is True across all configurations within each experiment ID for ACO and Q-learning
valid_aco_experiments = aco_random_data.groupby('Experiment ID').filter(lambda x: x['Find'].all())
valid_q_learning_experiments = q_learning_fixed_data.groupby('Experiment ID').filter(lambda x: x['Find'].all())

# Filtering data based on these valid experiments
aco_random_fixed_valid = valid_aco_experiments[valid_aco_experiments['E-tools'] == 'Fixed']['Travel Time Cost (seconds)']
aco_random_random_valid = valid_aco_experiments[valid_aco_experiments['E-tools'] == 'Random']['Travel Time Cost (seconds)']
q_learning_fixed_fixed_valid = valid_q_learning_experiments[valid_q_learning_experiments['E-tools'] == 'Fixed']['Travel Time Cost (seconds)']
q_learning_fixed_random_valid = valid_q_learning_experiments[valid_q_learning_experiments['E-tools'] == 'Random']['Travel Time Cost (seconds)']

# Plotting the aligned data
plt.figure(figsize=(12, 8))
plt.plot(aco_random_fixed_valid.reset_index(drop=True), label='ACO Random - Fixed E-tools', marker='o', linestyle='-', markersize=5)
plt.plot(aco_random_random_valid.reset_index(drop=True), label='ACO Random - Random E-tools', marker='x', linestyle='-', markersize=5)
plt.plot(q_learning_fixed_fixed_valid.reset_index(drop=True), label='Q-learning Fixed - Fixed E-tools', marker='s', linestyle='-', markersize=5)
plt.plot(q_learning_fixed_random_valid.reset_index(drop=True), label='Q-learning Fixed - Random E-tools', marker='^', linestyle='-', markersize=5)
plt.title('Aligned Travel Time Cost Grouped by E-tools')
plt.xlabel('Experiment Index')
plt.ylabel('Travel Time Cost (seconds)')
plt.legend()
plt.show()
