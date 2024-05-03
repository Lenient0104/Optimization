import numpy as np

# Data where the last column is the target variable and the rest are features
data = np.array([
    [10, 18, 21, 33, 45, 8, 12], [27, 28, 44, 48, 50, 7, 12], [14, 23, 39, 48, 50, 3, 12],
    [8, 19, 32, 41, 42, 9, 12], [5, 10, 19, 27, 30, 5, 6], [13, 20, 23, 27, 42, 5, 9], [2, 7, 21, 28, 45, 5, 11],
    [23, 24, 35, 37, 45, 9, 12], [13, 17, 18, 30, 46, 4, 9], [8, 13, 14, 24, 26, 1, 2], [23, 31, 37, 42, 48, 3, 7],
    [24, 27, 28, 30, 49, 1, 12], [3, 4, 9, 12, 20, 5, 6], [4, 7, 19, 20, 34, 2, 4], [2, 15, 17, 23, 36, 3, 8],
    [8, 11, 12, 16, 44, 4, 7], [13, 19, 30, 38, 46, 4, 12]
])

mean_values = np.mean(data, axis=0)
std_dev_values = np.std(data, axis=0)
min_values = np.min(data, axis=0)
max_values = np.max(data, axis=0)

num_new_rows = 1

# Generate random data
random_data = np.random.randn(num_new_rows, data.shape[1]) * std_dev_values + mean_values

# Sort the first five and the last two numbers in each row
for i in range(len(random_data)):
    random_data[i, :5] = np.sort(random_data[i, :5])  # Sort the first five numbers
    random_data[i, -2:] = np.sort(random_data[i, -2:])  # Sort the last two numbers

# Adjust the data within the min and max range
adjusted_data = np.clip(random_data, min_values, max_values)
new_data = adjusted_data.astype(int)  # Convert to integer

# Print the new data
for row in new_data:
    print(row.tolist())
