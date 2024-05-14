import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import json

def plot_speed_boxplot(json_files, vehicle_type):
    speeds = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            query_results = json.load(f)

        vehicle_speeds = []

        for result in query_results:
            speed = result[f'{vehicle_type}_speed']
            vehicle_speeds.append(float(speed) if speed != 'N/A' else None)

        speeds.append(vehicle_speeds)

    plt.figure(figsize=(10, 6))
    plt.boxplot(speeds, labels=simulation_times)
    plt.title(f'Boxplot of {vehicle_type.capitalize()} Speeds at Different Simulation Times')
    plt.xlabel('Simulation Time')
    plt.ylabel(f'{vehicle_type.capitalize()} Speed')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{vehicle_type}_speed_boxplot.png')
    plt.show()

# Plot boxplots for each vehicle type
simulation_times = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
json_files = [f'query_results-{sim}.json' for sim in simulation_times]
vehicle_types = ['pedestrian', 'bike', 'car']

for vehicle_type in vehicle_types:
    plot_speed_boxplot(json_files, vehicle_type)
