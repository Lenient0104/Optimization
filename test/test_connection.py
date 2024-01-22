import random

file_path = 'test.txt'

# A dictionary to store the 'to' node for each 'from' node
edges = {}

# Read the file line by line
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Extract the 'from' and 'to' values
        from_index = line.find('from="')
        to_index = line.find('to="')

        if from_index != -1 and to_index != -1:
            from_value = line[from_index + 6:line.find('"', from_index + 6)]
            to_value = line[to_index + 4:line.find('"', to_index + 4)]

            # Add the edge to the dictionary
            edges[from_value] = to_value


# Function to generate a path of a given length from a given start node
def generate_path(start_node, length):
    path = [start_node]
    current_node = start_node

    while len(path) < length and current_node in edges:
        current_node = edges[current_node]
        path.append(current_node)

    return path


# Number of series to generate
num_series = 10

# Desired lengths of the series
desired_lengths = [20, 10, 10, 6, 7, 10, 11, 12, 13, 14]

# Ensure that the desired_lengths list has length equal to num_series
if len(desired_lengths) != num_series:
    raise ValueError("The lengths list must have the same size as num_series")

# Generate the series
all_paths = []
for i in range(num_series):
    # Choose a random starting node
    start_node = random.choice(list(edges.keys()))

    # Generate a path of the desired length
    path = generate_path(start_node, desired_lengths[i])

    # Store the path
    all_paths.append(path)

# Display all generated paths
for i, path in enumerate(all_paths):
    print(', '.join(path))
