import pandas as pd

# Load the CSV file
file_path = 'ACO-fixed-tools.csv'
data = pd.read_csv(file_path)

# Add a new column 'E-tools' with all values set to 'Fixed'
data['User Preference'] = 'Prefer No E-car Option'

# Save the updated dataframe to a new CSV file
new_file_path = 'Updated_ACO.csv'
data.to_csv(new_file_path, index=False)


# Load the CSV file
file_path = 'ACO-noecar.csv'
data = pd.read_csv(file_path)

# Add a new column 'E-tools' with all values set to 'Fixed'
data['User Preference'] = 'No Preference'

# Save the updated dataframe to a new CSV file
new_file_path = 'Updated_ACO_1.csv'
data.to_csv(new_file_path, index=False)
