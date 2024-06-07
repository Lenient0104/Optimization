import pandas as pd

# Load the CSV file
file_path = 'ACO-fixed-tools.csv'
data = pd.read_csv(file_path)

# Add a new column 'E-tools' with all values set to 'Fixed'
data['E-tools'] = 'Random'

# Save the updated dataframe to a new CSV file
new_file_path = 'Updated_ACO-fixed.csv'
data.to_csv(new_file_path, index=False)


# Load the CSV file
file_path = 'ACO-random-tools.csv'
data = pd.read_csv(file_path)

# Add a new column 'E-tools' with all values set to 'Fixed'
data['E-tools'] = 'Fixed'

# Save the updated dataframe to a new CSV file
new_file_path = 'Updated_ACO-random.csv'
data.to_csv(new_file_path, index=False)
