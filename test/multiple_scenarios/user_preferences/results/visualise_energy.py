import pandas as pd

# Load the CSV files
aco_energy = pd.read_csv('ACO-energy.csv')
q_learning_energy = pd.read_csv('qLearning_energy.csv')

# Filter the data for more and less energy scenarios
aco_energy_less = aco_energy[aco_energy['Initial Energy'] == 0.5]['Travel Time Cost (seconds)'].reset_index(drop=True)
aco_energy_more = aco_energy[aco_energy['Initial Energy'] == 1]['Travel Time Cost (seconds)'].reset_index(drop=True)
q_learning_energy_less = q_learning_energy[q_learning_energy['Initial Energy'] == 0.5][
    'Travel Time Cost (seconds)'].reset_index(drop=True)
q_learning_energy_more = q_learning_energy[q_learning_energy['Initial Energy'] == 1][
    'Travel Time Cost (seconds)'].reset_index(drop=True)


# Define a function to calculate the percentage where Q-Learning time cost is greater than ACO
def calculate_percentage_greater(q_learning, aco):
    return ((q_learning > aco).mean() * 100).round(2)


# Calculate the required percentages
percentage_q_learning_less_greater = calculate_percentage_greater(q_learning_energy_less, aco_energy_less)
percentage_q_learning_more_greater = calculate_percentage_greater(q_learning_energy_more, aco_energy_more)

# Print the results
print("Percentage where Q-Learning Less Energy > ACO Less Energy:", percentage_q_learning_less_greater, "%")
print("Percentage where Q-Learning More Energy > ACO More Energy:", percentage_q_learning_more_greater, "%")
