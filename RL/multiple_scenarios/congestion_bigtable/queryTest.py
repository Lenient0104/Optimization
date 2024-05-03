from google.cloud import bigtable
from google.cloud.bigtable import row_filters
import json

# Project ID, BT instance,  BT table
project_id = "forward-venture-415212"
instance_id = "bt-dev"
table_id = "speed_table"
# Initialize BT client, instance, and table
client = bigtable.Client(project=project_id, admin=True)
instance = client.instance(instance_id)
table = instance.table(table_id)
column_family_id = 'traffic_info'


def query_speed_at_time(simulation_time):
    print(f"Querying speeds at simulation time: {simulation_time}")

    # Row key = {simulation_time}#{edge_id}
    # Construct the row key range for the given simulation time
    start_key = f"{simulation_time}#".encode()
    end_key = f"{simulation_time}#{{".encode()

    rows = table.read_rows(start_key=start_key, end_key=end_key)
    query_results = []

    for row in rows:
        row_key = row.row_key.decode()
        _, edge_id = row_key.split("#", 1)  # Split once since edge_id might contain '#'

        pedestrian_speed_cells = row.cells[column_family_id].get(b'pedestrian_speed', [])
        bike_speed_cells = row.cells[column_family_id].get(b'bike_speed', [])
        car_speed_cells = row.cells[column_family_id].get(b'car_speed', [])

        pedestrian_speed = pedestrian_speed_cells[0].value.decode('utf-8') if pedestrian_speed_cells else 'N/A'
        bike_speed = bike_speed_cells[0].value.decode('utf-8') if bike_speed_cells else 'N/A'
        car_speed = car_speed_cells[0].value.decode('utf-8') if car_speed_cells else 'N/A'

        query_results.append({
            "edge_id": edge_id,
            "simulation_time": simulation_time,
            "pedestrian_speed": pedestrian_speed,
            "bike_speed": bike_speed,
            "car_speed": car_speed
        })

    with open(f'query_results-{simulation_time}.json', 'w') as json_file:
        json.dump(query_results, json_file, indent=4)


simulation_time = 86400
query_speed_at_time(simulation_time)