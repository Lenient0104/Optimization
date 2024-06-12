import matplotlib.pyplot as plt
import numpy as np

# Data setup
main_scenarios = ["Energy Rate\n(Q-learning > ACO)", "E-Tools Distribution\n(Q-learning > ACO)", "User Preference\n(Q-learning > ACO)"]
sub_scenarios = [
    "More Energy", "Less Energy",
    "Fixed", "Random",
    "Default Preference", "Excluded E-car Option"
]
percentages = [4.8, 5.6, 8.9, 9.11, 7.05, 12.18]
colors = ['#89CFF0', '#7393B3',   # 莫兰迪蓝色系
          '#8F9779', '#4F7942',# 莫兰迪绿色系
          '#B57EDC', '#86608E']

# Setup for grouped bar chart
fig, ax = plt.subplots()
plt.rcParams.update({
    'font.size': 16,      # Global font size
    # 'axes.titlesize': 20, # Title font size
    'axes.labelsize': 16, # X and Y axis labels font size
    'xtick.labelsize': 16, # X tick labels font size
    'ytick.labelsize': 16, # Y tick labels font size
    # 'legend.fontsize': 18, # Legend font size
    # 'figure.titlesize': 22 # Figure title font size
})
index = np.arange(len(main_scenarios))  # the label locations
bar_width = 0.35  # the width of the bars

# Plotting
bars = []
for i, (perc, color) in enumerate(zip(percentages, colors)):
    offset = -bar_width/2 if i % 2 == 0 else bar_width/2
    bar = ax.bar(index[i // 2] + offset, perc, bar_width, color=color, label=sub_scenarios[i])
    bars.append(bar)

# Add labels and title
ax.set_xlabel('Main Scenarios', fontsize=16)
ax.set_ylabel('Percentage of OD Pairs (%)', fontsize=14)
ax.set_xticks(index)
ax.set_xticklabels(main_scenarios, fontsize=14)

# Create legend grouping labels by main scenario
handles, labels = ax.get_legend_handles_labels()
new_labels = []
new_handles = []
# Group labels by their main scenario
for i in range(0, len(labels), 2):
    new_labels.extend([labels[i], labels[i+1]])
    new_handles.extend([handles[i], handles[i+1]])
ax.legend(new_handles, new_labels, title="Sub-Scenarios", fontsize=14)

# Adding percentage values above each bar
for bar_group in bars:
    for bar in bar_group:
        height = bar.get_height()
        ax.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Set y-axis limit to make differences more pronounced
ax.set_ylim(0, max(percentages) + 5)
fig.set_size_inches(10, 6)
fig.tight_layout()
plt.savefig('multiple-scenarios.eps', format='eps')
plt.show()
