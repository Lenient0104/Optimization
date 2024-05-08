import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


stations = [5, 10, 15, 20]  # Different numbers of stations
algorithms = {
    'aco': 'ACO_results_antnum350_',
    'qlearning': 'Q_learning_results_episode400_',
    'dqn': 'DQN_experiment_results_episode400_'
}


# Function to process each algorithm's data for different station counts
def process_algorithm_data(algo_prefix, find_restriction=False):
    time_costs = []
    exec_times = []

    for station_count in stations:
        file_name = f"{algo_prefix}{station_count}stations.csv"
        data = pd.read_csv(file_name)

        # Renaming columns to standardize across different algorithm results
        if 'Execution Time' in data.columns:
            data.rename(columns={'Execution Time': 'Execution Time (seconds)'}, inplace=True)
        if 'Time Cost' in data.columns:
            data.rename(columns={'Time Cost': 'Time Cost (seconds)'}, inplace=True)

        if find_restriction and 'Find' in data.columns:
            data = data[data['Find'] == True]

        avg_time_cost = data['Time Cost (seconds)'].mean()
        avg_exec_time = data['Execution Time (seconds)'].mean()

        time_costs.append(avg_time_cost)
        exec_times.append(avg_exec_time)

    return time_costs, exec_times


# Process the data for each algorithm
aco_time_cost, aco_exec_time = process_algorithm_data(algorithms['aco'])
qlearning_time_cost, qlearning_exec_time = process_algorithm_data(algorithms['qlearning'])
dqn_time_cost, dqn_exec_time = process_algorithm_data(algorithms['dqn'], find_restriction=True)

# Create the plots for Total Travel Time Cost and Execution Time Cost
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Total Travel Time Cost
ax[0].plot(stations, aco_time_cost, marker='o', label='ACO')
ax[0].plot(stations, qlearning_time_cost, marker='s', label='Q-Learning')
ax[0].plot(stations, dqn_time_cost, marker='^', label='DQN')
ax[0].set_xlabel('Number of Stations')
ax[0].set_ylabel('Total Travel Time Cost (seconds)')
ax[0].set_title('Total Travel Time Cost by Number of Stations')
ax[0].set_xticks(stations)  # Set x-axis to show only the specific station numbers
ax[0].legend()

# Execution Time Cost
ax[1].plot(stations, aco_exec_time, marker='o', label='ACO')
ax[1].plot(stations, qlearning_exec_time, marker='s', label='Q-Learning')
ax[1].plot(stations, dqn_exec_time, marker='^', label='DQN')
ax[1].set_xlabel('Number of Stations')
ax[1].set_ylabel('Execution Time Cost (seconds)')
ax[1].set_title('Execution Time Cost by Number of Stations')
ax[1].set_xticks(stations)  # Set x-axis to show only the specific station numbers
ax[1].legend()

plt.tight_layout()
plt.show()
