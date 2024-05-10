import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('results-1.csv')

# Group the data by 'Episode'
grouped_data = data.groupby('Number of Ants')

# Get the episodes and corresponding execution times and time costs
episodes = list(grouped_data.groups.keys())
all_execution_times = [grouped_data.get_group(episode)['Execution Time (seconds)'] for episode in episodes]
all_time_costs = [grouped_data.get_group(episode)['Time Cost (seconds)'] for episode in episodes]

plt.rcParams.update({'font.size': 18})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Customize the boxplot appearance
boxprops = dict(linestyle='-', linewidth=1.5, color='black', facecolor='cornflowerblue')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkblue')
flierprops = dict(marker='o', color='black', alpha=0.5)

# Boxplot for Execution Time
bp1 = ax1.boxplot(all_execution_times, positions=episodes, widths=45, boxprops=boxprops,
                  medianprops=medianprops, flierprops=flierprops, patch_artist=True)
ax1.set_xticklabels(episodes, rotation=45, ha='right')
ax1.set_xlabel('Number of Ants', fontsize=18)
ax1.set_ylabel('Execution Time (seconds)', fontsize=18)
ax1.set_title('ACO Performance: Execution Time vs. Number of Ants', fontsize=19)
ax1.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

# Boxplot for Time Cost
bp2 = ax2.boxplot(all_time_costs, positions=episodes, widths=45, boxprops=boxprops,
                  medianprops=medianprops, flierprops=flierprops, patch_artist=True)
ax2.set_xticklabels(episodes, rotation=45, ha='right')
ax2.set_xlabel('Number of Ants', fontsize=18)
ax2.set_ylabel('Travel Time Cost (seconds)', fontsize=18)
ax2.set_title('ACO Performance: Travel Time Cost vs. Number of Episodes', fontsize=19)
ax2.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

plt.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.show()
