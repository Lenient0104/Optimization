import csv

def read_csv_column(filename, column_index):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            value = row[column_index]
            if value != 'inf':
                data.append(float(value))  # Convert the entry to numeric
    return data

def read_multiple_csv_files(filenames, column_index):
    all_data = []
    for filename in filenames:
        column_data = read_csv_column(filename, column_index)
        all_data.append(column_data)
    return all_data

filenames = ['ACO_results_100speednew.csv', 'ACO_results_80speednew.csv', 'ACO_results_60speednew.csv', 'ACO_results_40speednew.csv', 'ACO_results_20speednew.csv']  # List of filenames
column_index = 2 # Index of the column you want to read (0-based indexing)

column_data_1, column_data_2, column_data_3, column_data_4, column_data_5 = read_multiple_csv_files(filenames, column_index)
all_column_data = read_multiple_csv_files(filenames, column_index)

print("Column 1 data:", column_data_1)
print("Column 2 data:", column_data_2)


import matplotlib.pyplot as plt



plt.figure(figsize=(10, 8))
# Creating box plots
plt.boxplot(all_column_data)


# Specify the tick locations and labels
tick_locations = [1, 2, 3, 4, 5]
tick_labels = ['100% of speed', '80% of speed', '60% of speed', '40% of speed', '20% of Speed']
plt.xticks(tick_locations, tick_labels, rotation=45)
# Adding title and labels
plt.title('Box plots for column data')
plt.xlabel('Level of Congestion')
plt.ylabel('Time Cost')
plt.show()