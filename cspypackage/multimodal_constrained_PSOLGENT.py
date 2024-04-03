import networkx as nx
import numpy as np
from cspy import PSOLGENT
from networkx import DiGraph
from optimization1 import Optimization
from user_info import User
from numpy import zeros, ones, array

net_xml_path = 'DCC.net.xml'
source_edge = '361450282'
target_edge = "-110407380#1"
start_mode = 'walking'
user = User(60, True, 0, 20)
db_path = 'test_new.db'

# optimizer_interface = Optimization(net_xml_path, user, db_path, source_edge, target_edge)
# graph = optimizer_interface.new_graph
# print(graph)


# def build_min_time_graph(start_edge, destination_edge):
#     # First, build the initial paths graph using the existing method
#     paths_graph = graph
#
#     # Create a new DiGraph to store edges with the minimum time
#     min_time_graph = nx.DiGraph()
#
#     # Track the minimum time for each source-destination pair
#     min_times = {}
#
#     for source, target, data in paths_graph.edges(data=True):
#         edge_time = data['weight']  # Assuming 'weight' is the total time
#         edge_key = (source, target)
#
#         if edge_key not in min_times or edge_time < min_times[edge_key]['weight']:
#             min_times[edge_key] = data  # Update with the faster path data
#
#     # Add nodes and edges with the minimum time to the new graph
#     for (source, target), data in min_times.items():
#         min_time_graph.add_node(source)
#         min_time_graph.add_node(target)
#         min_time_graph.add_edge(source, target, **data)
#
#     # Now min_time_graph contains only the edges with the smallest time between each pair of nodes
#     print("The graph edges are:", min_time_graph.edges)
#     # structure of the min_time_graph:
#     # weight=total_time, key='walking', path=path, pheromone_level=0.1, distance=distance
#     return min_time_graph

def calculate_energy_comsumption(current_mode, distance):
    if current_mode == 'walking':
        return 0
    # Define vehicle efficiency in Wh per meter (converted from Wh per km)
    vehicle_efficiency = {'e_bike_1': 20 / 1000, 'e_scooter_1': 25 / 1000, 'e_car': 150 / 1000}
    # battery_capacity = {'e_bike_1': 500, 'e_scooter_1': 250, 'e_car': 50000}
    battery_capacity = {'e_bike_1': 500, 'e_scooter_1': 250, 'e_car': 5000}
    energy_consumed = vehicle_efficiency[current_mode] * distance
    # Calculate the delta SoC (%) for the distance traveled
    delta_soc = (energy_consumed / battery_capacity[current_mode]) * 100

    return delta_soc


def build_multimodal_graph(paths_graph):
    multimodal_graph = nx.DiGraph()
    print("Building multimodal graph...")

    for source, target, data in paths_graph.edges(data=True):
        mode = data.get('mode')
        total_time = data.get('weight')
        distance = data.get('distance')

        print(f"Adding edge from {source} to {target}, mode={mode}, time={total_time}, distance={distance}")

        virtual_node = f"{source}_{target}_{mode}"
        multimodal_graph.add_edge(source, virtual_node, weight=total_time, mode=mode, distance=distance)
        multimodal_graph.add_edge(virtual_node, target, weight=0.001, mode=mode)

    print(f"Graph built with {len(multimodal_graph.nodes())} nodes and {len(multimodal_graph.edges())} edges.")
    return multimodal_graph



def build_energy_cost_graph(multimodal_graph, source_edge, target_edge):
    G = nx.DiGraph(directed=True, n_res=1)  # For each mode, the electricity is the only constraint currently

    for source, target, data in multimodal_graph.edges(data=True):
        mode = data.get('mode')
        time_cost = data.get('weight')

        # 判断是否为虚拟节点
        if 'virtual' in source or 'virtual' in target:
            # 虚拟节点到实际节点的边，权重应该很小，几乎不影响总体成本
            energy_cost = 0
        else:
            distance = data.get('distance', 0)
            # 计算能耗成本
            energy_cost = calculate_energy_comsumption(mode, distance) if distance > 0 else 0
            print(energy_cost)

        # 处理起始点和终点
        mapped_source = 'Source' if source == source_edge else source
        mapped_target = 'Sink' if target == target_edge else target

        # 添加边到图中，包含时间成本和能耗成本
        G.add_edge(mapped_source, mapped_target, res_cost=np.array([energy_cost]), weight=time_cost)

    return G


if __name__ == '__main__':
    optimizer_interface = Optimization(net_xml_path, user, db_path, source_edge, target_edge)
    original_graph = optimizer_interface.build_new_graph(source_edge, target_edge)  # 假设这是原始图构建方法
    print(original_graph)
    # multimodal_graph = build_multimodal_graph(original_graph)
    # print(multimodal_graph)
    # G = build_energy_cost_graph(multimodal_graph, source_edge, target_edge)
    # print(G)
    #
    # n_nodes = len(G.nodes())
    # psolgent = PSOLGENT(G, [100], [0], max_iter=200, swarm_size=50, member_size=n_nodes, neighbourhood_size=50)
    # psolgent.run()
    #
    # total_time_cost = 0  # Initialize the total time cost
    #
    # print("Optimal path:")
    # path = psolgent.path
    # for i in range(len(path) - 1):
    #     source = path[i]
    #     target = path[i + 1]
    #
    #     # Skip virtual nodes when calculating total time cost
    #     if 'virtual' not in source and 'virtual' not in target:
    #         # Get the actual source and target names for the edge in the original graph
    #         actual_source = source_edge if source == 'Source' else source
    #         actual_target = target_edge if target == 'Sink' else target
    #
    #         # Find the edge data in the original graph to get mode and time cost
    #         data = original_graph.get_edge_data(actual_source, actual_target)
    #         mode = data['mode'] if 'mode' in data else 'walking'  # Default to walking if no mode is specified
    #         time_cost = data['weight']
    #
    #         total_time_cost += time_cost  # Accumulate the total time cost
    #
    #         print(f"From {actual_source} to {actual_target}: Mode={mode}, Time Cost={time_cost}")
    #
    # print(f"Total time cost: {total_time_cost}")

