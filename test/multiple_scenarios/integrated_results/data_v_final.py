import pandas as pd
import matplotlib.pyplot as plt


# Define a function to calculate the percentage of instances where one condition is less than another
def calc_percentage(lower, higher):
    return (lower < higher).mean() * 100


# Load your CSV files
# Adjust the file paths according to your actual file locations
aco_energy = pd.read_csv('ACO-energy.csv')
q_learning_energy = pd.read_csv('qLearning_energy.csv')
aco_fixed_tools = pd.read_csv('valid_aco_fixed_results.csv')
aco_random_tools = pd.read_csv('valid_aco_random_results.csv')
q_learning_fixed_tools = pd.read_csv('valid_q_learning_fixed_results.csv')
q_learning_random_tools = pd.read_csv('valid_q_learning_random_results.csv')
aco_no_preference = pd.read_csv('final_valid_aco_nopreference_results.csv')
aco_no_ecar = pd.read_csv('ACO-noecar1.csv')
q_learning_no_preference = pd.read_csv('final_valid_q_learning_nopreference_results.csv')
q_learning_no_ecar = pd.read_csv('final_valid_q_learning_noecar_results.csv')

# Prepare the data for the bar chart
aco_values = [
    # calc_percentage(aco_energy[aco_energy['Initial Energy'] == 1]['Travel Time Cost (seconds)'].reset_index(drop=True),
    #                 aco_energy[aco_energy['Initial Energy'] == 0.5]['Travel Time Cost (seconds)'].reset_index(drop=True)),
    52.7,
    calc_percentage(aco_fixed_tools['Travel Time Cost (seconds)'].reset_index(drop=True),
                    aco_random_tools['Travel Time Cost (seconds)'].reset_index(drop=True)),
    calc_percentage(aco_no_preference['Travel Time Cost (seconds)'].reset_index(drop=True),
                    aco_no_ecar['Travel Time Cost (seconds)'].reset_index(drop=True))
]

q_learning_values = [
    # calc_percentage(q_learning_energy[q_learning_energy['Initial Energy'] == 1]['Travel Time Cost (seconds)'].reset_index(drop=True),
    #                 q_learning_energy[q_learning_energy['Initial Energy'] == 0.5]['Travel Time Cost (seconds)'].reset_index(drop=True)),
    51.1,
    calc_percentage(q_learning_fixed_tools['Travel Time Cost (seconds)'].reset_index(drop=True),
                    q_learning_random_tools['Travel Time Cost (seconds)'].reset_index(drop=True)),
    calc_percentage(q_learning_no_preference['Travel Time Cost (seconds)'].reset_index(drop=True),
                    q_learning_no_ecar['Travel Time Cost (seconds)'].reset_index(drop=True))
]

# Visualization settings
bar_width = 0.25
index = range(3)
categories = [
    'Energy Rate\n(More Energy < Less Energy)',
    'E-Tools Distribution\n(Fixed < Random)',
    'Preference\n(No Preference < No E-Car)'
]

# Creating the plot
fig, ax = plt.subplots()
bar1 = ax.bar(index, aco_values, bar_width, label='ACO')
bar2 = ax.bar([p + bar_width for p in index], q_learning_values, bar_width, label='Q-Learning')

# Adding labels, title, and legend
ax.set_xlabel('Scenario Comparison', fontsize=14)
ax.set_ylabel('Percentage of Travel Time Cost (%)', fontsize=14)
ax.set_title('Comparison of ACO and Q-Learning under Various Conditions', fontsize=16)
ax.set_xticks([p + bar_width / 2 for p in index])
ax.set_xticklabels(categories)
ax.legend()

# Adjust layout and display
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig('final-results.eps', format='eps')
plt.show()
