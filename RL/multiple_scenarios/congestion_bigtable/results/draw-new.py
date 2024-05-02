import pandas as pd
import matplotlib.pyplot as plt

# Load the data from all three files
file_aco = "ACO-results-simulation_time.csv"
file_q_learning = "Q_learning_results_simulation.csv"
file_dqn = "DQN_experiment_results_simulation.csv"


# Read the CSV files into DataFrame
df_aco = pd.read_csv(file_aco)
df_q_learning = pd.read_csv(file_q_learning)
df_dqn = pd.read_csv(file_dqn)

# Renaming the columns to be uniform across datasets for easier manipulation
df_aco.rename(columns={"Travel Time Cost (seconds)": "Travel Time Cost"}, inplace=True)
df_q_learning.rename(columns={"Travel Time Cost (seconds)": "Travel Time Cost"}, inplace=True)
df_dqn.rename(columns={"Travel Time Cost (seconds)": "Travel Time Cost"}, inplace=True)

# Adding a column to each dataframe to identify the algorithm
df_aco['Algorithm'] = 'ACO'
df_q_learning['Algorithm'] = 'Q-Learning'
df_dqn['Algorithm'] = 'DQN'

# Filter out rows where Find is False
df_aco = df_aco[df_aco['Find'] == True]
df_q_learning = df_q_learning[df_q_learning['Find'] == True]
df_dqn = df_dqn[df_dqn['Find'] == True]

# Combine the dataframes into a single dataframe
combined_df = pd.concat([df_aco, df_q_learning, df_dqn], ignore_index=True)
combined_df = combined_df[['Experiment ID', 'Simulation Time', 'Travel Time Cost', 'Algorithm']]

# Group by Simulation Time and Experiment ID to find the algorithm with the minimum travel time per OD pair
best_algo_per_od_time = combined_df.groupby(['Simulation Time', 'Experiment ID']).apply(
    lambda group: group[group['Travel Time Cost'] == group['Travel Time Cost'].min()]
)
best_algo_per_od_time.reset_index(drop=True, inplace=True)

# Count how many times each algorithm is the best for each simulation time
best_algo_count = best_algo_per_od_time.groupby(['Simulation Time', 'Algorithm']).size().unstack(fill_value=0)

# Plotting the data
fig, ax = plt.subplots(figsize=(14, 8))
best_algo_count.plot(kind='bar', ax=ax)
ax.set_title('Best Performing Algorithm for Each Simulation Time', fontsize=16)
ax.set_xlabel('Simulation Time', fontsize=14)
ax.set_ylabel('Count of Best Performance', fontsize=14)
ax.legend(title='Algorithm')
plt.xticks(rotation=0)
plt.show()
