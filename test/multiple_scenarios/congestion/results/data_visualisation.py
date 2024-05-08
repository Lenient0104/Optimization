import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

algorithms = {
    'ACO': 'ACO_results_antnum350_',
    'Q_learning': 'Q_learning_results_episode400_',
    'DQN': 'DQN_experiment_results_episode400_'
}
conditions = ['normal', 'congested']  # Traffic conditions


# Function to load and process data for an algorithm under specific traffic conditions
def load_and_process_data(algo_prefix):
    results = {}
    for condition in conditions:
        file_name = f"{algo_prefix}{condition}.csv"
        data = pd.read_csv(file_name)

        if 'Time Cost (seconds)' in data.columns:
            data.rename(columns={'Time Cost (seconds)': 'Travel Time Cost (seconds)'}, inplace=True)

        # Assuming 'Find' column exists and we need to filter by True only
        if 'Find' in data.columns:
            data = data[data['Find']]

        # Store data in a dictionary by condition
        results[condition] = data
    return results


# Function to balance the number of data points by padding with the mean
def pad_with_mean(data, target_length):
    current_length = len(data)
    if current_length < target_length:
        mean_value = data.mean()
        # Create additional data points to reach the target length
        additional_data = np.random.choice(data, target_length - current_length, replace=True)
        padded_data = np.concatenate([data, additional_data])
    else:
        padded_data = data
    return padded_data


# Set up the plotting
fig, axs = plt.subplots(nrows=len(algorithms), ncols=2, figsize=(12, 18), constrained_layout=True)
colors = {'normal': '#1f77b4', 'congested': '#ff7f0e'}  # Blue and orange

max_length = 0  # Variable to store the maximum number of data points across all conditions and algorithms

# First, determine the maximum number of trials among all conditions for all algorithms
for algo_prefix in algorithms.values():
    for condition in conditions:
        data = pd.read_csv(f"{algo_prefix}{condition}.csv")
        if 'Find' in data.columns:
            data = data[data['Find']]
        max_length = max(max_length, len(data))

# Mapping to give each algorithm its own plot for each metric
for i, (algo_name, algo_prefix) in enumerate(algorithms.items()):
    data = load_and_process_data(algo_prefix)
    for j, metric in enumerate(['Travel Time Cost (seconds)', 'Execution Time (seconds)']):
        for k, condition in enumerate(conditions):
            results = data[condition][metric]
            # Pad data with mean if necessary
            padded_results = pad_with_mean(results, max_length)
            # Plotting the bars
            x_positions = np.arange(len(padded_results)) + j * (max_length + 1)
            axs[i, j].bar(x_positions, padded_results, color=colors[condition], label=f'{condition}', alpha=0.7)

        # Add a custom title below the subplot using text
        axs[i, j].text(0.5, -0.03, f'{algo_name} - {metric.split()[0]} Cost', transform=axs[i, j].transAxes,
                       horizontalalignment='center', fontsize=14)
        axs[i, j].set_xticks([])
        axs[i, j].set_ylabel('Seconds')
        axs[i, j].legend()
        # axs[i, j].set_title(f'{algo_name} - {metric.split()[0]} Cost')

plt.suptitle('Individual Trial Performance Under Different Traffic Conditions', fontsize=20)
plt.show()
