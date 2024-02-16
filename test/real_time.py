import xml.etree.ElementTree as ET
import networkx as nx
import time
import pandas as pd

# Initialize your graph
G = nx.DiGraph()

# Read the CSV data into a DataFrame
df = pd.read_csv('../aco/optimization/graph202308142106_edges.csv')  # Replace 'your_data.csv' with your actual CSV file name


def fetch_realtime_data(step):
    # Filter the DataFrame based on the 'step' column
    filtered_data = df[df['step'] == step]

    # Convert the filtered DataFrame to a list of dictionaries
    return filtered_data.to_dict('records')




def update_graph(G, data):
    for item in data:
        edge_id = item['edge_id']
        mean_speed = item['mean_speed']
        G.add_edge(edge_id + '_start', edge_id + '_end', weight=1 / mean_speed)


# Start from step 1
current_step = 1

while True:
    # Fetch new real-time data for the current step
    data = fetch_realtime_data(current_step)

    # Update the graph with new data
    update_graph(G, data)

    # Assume you want to find the route between edge '-1' and edge '0'
    start_edge = '-1'
    end_edge = '0'

    try:
        path = nx.shortest_path(G, source=start_edge + '_start', target=end_edge + '_end', weight='weight')
        recommended_route = [edge.replace('_start', '').replace('_end', '') for edge in path]
        print(f"Step {current_step} - Recommended Route:", recommended_route)
    except nx.NetworkXNoPath:
        print(f"Step {current_step} - No available path between the given edges.")

    # Increase step
    current_step += 1

    # Pause for a while before fetching new data
    time.sleep(5)
