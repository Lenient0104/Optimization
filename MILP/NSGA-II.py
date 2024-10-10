import re
import json
from tqdm import tqdm
import random as rn
import numpy as np
import networkx as nx
import xml.etree.ElementTree as ET
# import matplotlib
# matplotlib.use('macOSX')  # 或者 'Qt5Agg', 'macOSX' 等，根据您的系统环境选择一个合适的后端
import matplotlib.pyplot as plt
import math

class GraphHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.G = self.create_graph_from_net_xml()

    def create_graph_from_net_xml(self):
        unique_edges = []
        connections = []
        pattern = r"^[A-Za-z]+"
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        edge_detail_map = {}
        for edge in tqdm(root.findall('edge'), desc="Processing edges"):
            for lane in edge.findall('lane'):
                edge_id = edge.attrib['id']
                edge_detail_map[edge_id] = {
                    'length': float(lane.attrib['length']),
                    'speed_limit': float(lane.attrib['speed']),
                    'shape': lane.attrib['shape'],
                }

        for conn in tqdm(root.findall('connection'), desc="Processing connections", mininterval=1.0):
            pairs = []
            from_edge = conn.get('from')
            to_edge = conn.get('to')

            if from_edge.startswith(":") or re.match(pattern, from_edge) is not None:
                continue
            if from_edge not in unique_edges and from_edge != 'gneE29':
                unique_edges.append(from_edge)
            if to_edge not in unique_edges:
                unique_edges.append(to_edge)

            pairs.append(from_edge)
            pairs.append(to_edge)
            connections.append(pairs)

        G = nx.DiGraph()
        for edge in unique_edges:
            G.add_node(edge, **edge_detail_map.get(edge, {}))

        for from_edge, to_edge in connections:
            if from_edge in edge_detail_map and to_edge in edge_detail_map:
                length = edge_detail_map[from_edge]['length']
                G.add_edge(from_edge, to_edge, weight=length)

        return G

    def get_graph(self):
        return self.G


class PreferenceGenerator:
    def __init__(self, G, station_types):
        self.G = G  # original graph
        self.station_types = station_types  # station_types = ['eb', 'es', 'ec', 'walk']
        self.node_station_pair = {}

    def generate_node_preferences(self):
        preferred_station = {}
        preferred_nodes = ['361409608#3', '3791905#2', '-11685016#2', '369154722#2', '244844370#0', '37721356#0',
                           '74233405#1', '129774671#0', '23395388#5', '-64270141', '18927706#0', '-42471880',
                           '67138626#1', '41502636#0', '-75450412', '-23347664#1', '14151839#3', '-12341242#1',
                           '-13904652',
                           '-47638297#3']

        for i in self.G.nodes:
            if i in preferred_nodes:
                preferred_types = ['eb', 'es', 'ec', 'walk']
                preferred_station[i] = preferred_types  # random.sample(preferred_types, num_preferred)
                if 'walk' not in preferred_station[i]:
                    preferred_station[i].append('walk')
                self.node_station_pair[i] = preferred_station[i]
            else:
                preferred_station[i] = ['']

        return preferred_station, preferred_nodes

class ShortestPathComputer:
    def __init__(self, graph):
        self.graph = graph

    def compute_shortest_paths_start(self, start_node, preference_stations):
        shortest_routes_start = {}
        for station in preference_stations:
            try:
                shortest_path = nx.shortest_path(self.graph, source=start_node, target=station, weight='weight')
                shortest_routes_start[station] = (
                    shortest_path,
                    nx.shortest_path_length(self.graph, source=start_node, target=station, weight='weight'))
            except nx.NetworkXNoPath:
                pass
        return shortest_routes_start

    def compute_shortest_paths_pairs(self, preference_stations):
        all_shortest_routes_pairs = {}
        for station1 in preference_stations:
            for station2 in preference_stations:
                if station1 != station2:
                    try:
                        shortest_path = nx.shortest_path(self.graph, source=station1, target=station2, weight='weight')
                        all_shortest_routes_pairs[(station1, station2)] = (shortest_path,
                                                                           nx.shortest_path_length(self.graph,
                                                                                                   source=station1,
                                                                                                   target=station2,
                                                                                                   weight='weight'))
                    except nx.NetworkXNoPath:
                        pass
        return all_shortest_routes_pairs

    def compute_shortest_paths_dest(self, dest_node, preference_stations):
        shortest_routes_dest = {}
        for station in preference_stations:
            try:
                shortest_path = nx.shortest_path(self.graph, source=station, target=dest_node, weight='weight')
                shortest_routes_dest[station] = (
                    shortest_path,
                    nx.shortest_path_length(self.graph, source=station, target=dest_node, weight='weight'))
            except nx.NetworkXNoPath:
                pass
        return shortest_routes_dest

    def compute_shortest_path_start_end(self, start_node, dest_node):
        try:
            shortest_path = nx.shortest_path(self.graph, source=start_node, target=dest_node, weight='weight')
            shortest_route_start_end = (
                shortest_path,
                nx.shortest_path_length(self.graph, source=start_node, target=dest_node, weight='weight'))
            return shortest_route_start_end
        except nx.NetworkXNoPath:
            return None


class ReducedGraphCreator:
    def __init__(self, graph, start_node, dest_node, preference_stations, shortest_routes_start, shortest_routes_dest,
                 all_shortest_routes_pairs):
        self.graph = graph
        self.start_node = start_node
        self.dest_node = dest_node
        self.preference_stations = preference_stations
        self.shortest_routes_start = shortest_routes_start
        self.shortest_routes_dest = shortest_routes_dest
        self.all_shortest_routes_pairs = all_shortest_routes_pairs
        # self.shortest_route_start_end = shortest_route_start_end

    def create_new_graph(self):
        new_graph = nx.DiGraph()

        new_graph.add_nodes_from([self.start_node, self.dest_node] + self.preference_stations)

        for station in self.preference_stations:
            try:
                new_graph.add_edge(station, self.dest_node, weight=self.shortest_routes_dest[station][1])
            except KeyError:
                pass

        for station, (shortest_path, cumulative_weight) in self.shortest_routes_start.items():
            try:
                new_graph.add_edge(self.start_node, station, weight=cumulative_weight)
            except KeyError:
                pass

        for (station1, station2), (shortest_path, cumulative_weight) in self.all_shortest_routes_pairs.items():
            try:
                new_graph.add_edge(station1, station2, weight=cumulative_weight)
            except KeyError:
                pass

        # if self.shortest_route_start_end is not None:
        #     new_graph.add_edge(self.start_node, self.dest_node, weight=self.shortest_route_start_end[1])

        return new_graph

# MINIMIZATION

# Initialize random population of parent chormosomes/solutions P
def random_population(graph, source, target, mode_options, n_sol):
    pop = []
    for _ in range(n_sol):
        try:
            path = nx.shortest_path(graph, source=source, target=target, weight='weight')
            mode_path = [rn.choice(mode_options) for _ in range(len(path) - 1)]
            pop.append((path, mode_path))
        except nx.NetworkXNoPath:
            # Handle the case where no path exists
            pop.append(([], []))
    return pop


# On each iteration, out of 2 randomly selected parents we create 2 offsprings
# by taking fraction of genes from one parent and remaining fraction from other parent
def crossover(pop, crossover_rate):
    offspring = []
    for _ in range(crossover_rate):
        r1 = np.random.randint(0, len(pop))
        r2 = np.random.randint(0, len(pop))
        while r1 == r2:
            r2 = np.random.randint(0, len(pop))

        # Combine parts of two paths and modes
        path1, mode_path1 = pop[r1]
        path2, mode_path2 = pop[r2]
        common_nodes = set(path1) & set(path2)

        if common_nodes:
            cut_node = rn.choice(list(common_nodes))
            index1 = path1.index(cut_node)
            index2 = path2.index(cut_node)

            new_path = path1[:index1] + path2[index2:]
            new_mode_path = mode_path1[:index1] + mode_path2[index2:]

            offspring.append((new_path, new_mode_path))
        else:
            offspring.append((path1, mode_path1))  # If no common nodes, keep original path
    return offspring


def mutation(pop, graph, mutation_rate, mode_options):
    offspring = []
    for i in range(mutation_rate):
        r = np.random.randint(0, len(pop))
        mutated_path, mutated_mode_path = pop[r]

        if len(mutated_path) > 2:
            mutation_point = np.random.randint(1, len(mutated_path) - 1)
            neighbors = list(graph.neighbors(mutated_path[mutation_point]))
            if neighbors:
                new_node = np.random.choice(neighbors)
                mutated_path[mutation_point] = new_node

                # Also mutate the mode randomly
                mutated_mode_path[mutation_point - 1] = rn.choice(mode_options)

        offspring.append((mutated_path, mutated_mode_path))
    return offspring


# On each iteration, out of 2 randomly selected parents we create 2 offsprings
# by excahging some amount of genes/coordinates between parents
def mutation(pop, mutation_rate):
    offspring = np.zeros((mutation_rate, pop.shape[1]))
    for i in range(int(mutation_rate / 2)):
        r1 = np.random.randint(0, pop.shape[0])
        r2 = np.random.randint(0, pop.shape[0])
        while r1 == r2:
            r1 = np.random.randint(0, pop.shape[0])
            r2 = np.random.randint(0, pop.shape[0])
        # We select only one gene/coordinate per chromosomes/solution for mutation here.
        # For binary solutions, number of genes for mutation can be arbitrary
        cutting_point = np.random.randint(0, pop.shape[1])
        offspring[2 * i] = pop[r1]
        offspring[2 * i, cutting_point] = pop[r2, cutting_point]
        offspring[2 * i + 1] = pop[r2]
        offspring[2 * i + 1, cutting_point] = pop[r1, cutting_point]

    return offspring  # arr(mutation_size x n_var)


# Create some amount of offsprings Q by adding fixed coordinate displacement to some
# randomly selected parent's genes/coordinates
def local_search(pop, n_sol, step_size):
    # number of offspring chromosomes generated from the local search
    offspring = np.zeros((n_sol, pop.shape[1]))
    for i in range(n_sol):
        r1 = np.random.randint(0, pop.shape[0])
        chromosome = pop[r1, :]
        r2 = np.random.randint(0, pop.shape[1])
        chromosome[r2] += np.random.uniform(-step_size, step_size)
        if chromosome[r2] < lb[r2]:
            chromosome[r2] = lb[r2]
        if chromosome[r2] > ub[r2]:
            chromosome[r2] = ub[r2]

        offspring[i, :] = chromosome
    return offspring  # arr(loc_search_size x n_var)


# Calculate fitness (obj function) values for each chormosome/solution
# Kursawe function - https://en.wikipedia.org/wiki/Test_functions_for_optimization
def evaluation(graph, pop, speed_dict, cost_coefficients, profit_margins):
    fitness_values = []

    for path, mode_path in pop:
        if path:
            total_time = 0
            total_cost = 0

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                mode = mode_path[i]
                edge_weight = graph[u][v]['weight']
                speed = speed_dict.get(u, {}).get(f'{mode}_speed', 1)
                time = edge_weight / speed
                total_time += time

                base_cost = cost_coefficients[mode] * edge_weight
                total_cost += base_cost * (1 + profit_margins[mode])

            fitness_values.append([total_time, total_cost])
        else:
            fitness_values.append([float('inf'), float('inf')])  # Invalid paths have infinite cost/time

    return np.array(fitness_values)


# Estimate how tightly clumped fitness values are on Pareto front.
def crowding_calculation(fitness_values):
    pop_size = len(fitness_values[:, 0])
    fitness_value_number = len(fitness_values[0, :])  # == n of objective functions
    matrix_for_crowding = np.zeros((pop_size, fitness_value_number))  # arr(pop_size x 2)
    normalized_fitness_values = (fitness_values - fitness_values.min(0)) / fitness_values.ptp(
        0)  # arr.ptp(0) array of max elem in each col

    for i in range(fitness_value_number):
        crowding_results = np.zeros(pop_size)
        crowding_results[0] = 1  # extreme point has the max crowding distance
        crowding_results[pop_size - 1] = 1  # extreme point has the max crowding distance
        sorted_normalized_fitness_values = np.sort(normalized_fitness_values[:, i])
        sorted_normalized_values_index = np.argsort(normalized_fitness_values[:, i])
        # crowding distance calculation. Say for fitness1[i], crowding = fitness1[i+1] - fitness1[i-1]
        crowding_results[1:pop_size - 1] = (
                    sorted_normalized_fitness_values[2:pop_size] - sorted_normalized_fitness_values[0:pop_size - 2])
        re_sorting = np.argsort(sorted_normalized_values_index)
        matrix_for_crowding[:, i] = crowding_results[re_sorting]

    crowding_distance = np.sum(matrix_for_crowding,
                               axis=1)  # on fitness1 - fitness2 plot, each point on pareto front has crowding distance number

    return crowding_distance  # arr(pop_size,)


# Crowding distance is used to maintain diversity of solutions on Pareto front.
# Remove some amount of solutions that are clumped together to much
def remove_using_crowding(fitness_values, number_solutions_needed):
    pop_index = np.arange(fitness_values.shape[0])
    crowding_distance = crowding_calculation(fitness_values)
    selected_pop_index = np.zeros(number_solutions_needed)
    selected_fitness_values = np.zeros((number_solutions_needed, len(fitness_values[0, :])))  # arr(num_sol_needed x 2)
    for i in range(number_solutions_needed):
        pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, pop_size - 1)
        solution_2 = rn.randint(0, pop_size - 1)
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            # solution 1 is better than solution 2
            selected_pop_index[i] = pop_index[solution_1]
            selected_fitness_values[i, :] = fitness_values[solution_1, :]
            pop_index = np.delete(pop_index, (solution_1), axis=0)
            fitness_values = np.delete(fitness_values, (solution_1), axis=0)
            crowding_distance = np.delete(crowding_distance, (solution_1), axis=0)
        else:
            # solution 2 is better than solution 1
            selected_pop_index[i] = pop_index[solution_2]
            selected_fitness_values[i, :] = fitness_values[solution_2, :]
            pop_index = np.delete(pop_index, (solution_2), axis=0)
            fitness_values = np.delete(fitness_values, (solution_2), axis=0)
            crowding_distance = np.delete(crowding_distance, (solution_2), axis=0)

    selected_pop_index = np.asarray(selected_pop_index, dtype=int)

    return selected_pop_index  # arr(n_sol_needed,)


# find indices of solutions that dominate others
def pareto_front_finding(fitness_values, pop_index):
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool)  # all True initially
    for i in range(pop_size):
        for j in range(pop_size):
            if all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                pareto_front[i] = 0  # i is not in pareto front becouse j dominates i
                break

    return pop_index[pareto_front]  # arr(len_pareto_front,)


# repeat Pareto front selection to build a population within defined size limits
def selection(pop, fitness_values, pop_size):
    pop_index_0 = np.arange(pop.shape[0])
    pareto_front_index = []

    while len(pareto_front_index) < pop_size:
        new_pareto_front = pareto_front_finding(fitness_values[pop_index_0, :], pop_index_0)
        total_pareto_size = len(pareto_front_index) + len(new_pareto_front)

        if total_pareto_size > pop_size:
            number_solutions_needed = pop_size - len(pareto_front_index)
            selected_solutions = remove_using_crowding(fitness_values[new_pareto_front], number_solutions_needed)
            new_pareto_front = new_pareto_front[selected_solutions]

        pareto_front_index = np.hstack((pareto_front_index, new_pareto_front))
        remaining_index = set(pop_index_0) - set(pareto_front_index)
        pop_index_0 = np.array(list(remaining_index))

    selected_pop = pop[pareto_front_index.astype(int)]
    return selected_pop


# Parameters
n_var = 3
lb = [-5, -5, -5]
ub = [5, 5, 5]
pop_size = 50
rate_crossover = 20
rate_mutation = 20
maximum_generation = 150
mode_options = ['eb', 'es', 'ec', 'walk']
cost_coefficients = {'eb': 0.01, 'es': 0.02, 'ec': 0.05, 'walk': 0}
profit_margins = {'eb': 0.10, 'es': 0.12, 'ec': 0.15, 'walk': 0}
file_path = "DCC.net.xml"
speed_file_path = 'query_results-0.json'
od_pairs_file = 'od_pairs.csv'  # Path to the CSV file containing OD pairs
output_csv_file = 'RG_TimeTest_50_Nodes.csv'  # Output CSV file to store the results

# Create graph from XML file
graph_handler = GraphHandler(file_path)
original_G = graph_handler.get_graph()

# Define parameters
num_nodes = len(original_G.nodes)
# User preferences
user_preference = ['eb', 'ec', 'es']
station_types = ['eb', 'es', 'ec', 'walk']
node_stations = {i: station_types for i in original_G.nodes}
start_node = '361450282'
end_node = '-110407380#1'
node_stations[start_node] = ['walk']
node_stations[end_node] = ['walk']
no_pref_nodes = 10
# Generate preferred station types for each node (execute only once)
preference_generator = PreferenceGenerator(original_G, station_types)
preferred_station, preferred_nodes = preference_generator.generate_node_preferences()

# Compute shortest routes pairs (execute only once)
shortest_path_computer = ShortestPathComputer(original_G)
# all_shortest_routes_pairs = shortest_path_computer.compute_shortest_paths_pairs(preferred_nodes)

# Load speed data from JSON
with open(speed_file_path, 'r') as f:
    speed_data = json.load(f)

# Create a dictionary for speed data
speed_dict = {entry['edge_id']: {'pedestrian_speed': float(entry['pedestrian_speed']),
                                 'bike_speed': float(entry['bike_speed']),
                                 'car_speed': float(entry['car_speed'])}
              for entry in speed_data}



all_shortest_routes_pairs = shortest_path_computer.compute_shortest_paths_pairs(preferred_nodes)
shortest_routes_start = shortest_path_computer.compute_shortest_paths_start(start_node, preferred_nodes)
shortest_routes_dest = shortest_path_computer.compute_shortest_paths_dest(end_node, preferred_nodes)
reduced_graph_creator = ReducedGraphCreator(original_G, start_node, end_node, preferred_nodes,
                                            shortest_routes_start, shortest_routes_dest,
                                            all_shortest_routes_pairs)
# reduced_graph_creator = ReducedGraphCreator(original_G, start_node, end_node, preferred_nodes,
#                                             shortest_routes_start, shortest_routes_dest,
#                                             all_shortest_routes_pairs, shortest_route_start_end)
reduced_G = reduced_graph_creator.create_new_graph()

# Random population initialization
pop = random_population(reduced_G, 'start_node', 'end_node', mode_options, pop_size)

# NSGA-II main loop
for i in range(maximum_generation):
    offspring_from_crossover = crossover(pop, reduced_G, rate_crossover)
    offspring_from_mutation = mutation(pop, reduced_G, rate_mutation, mode_options)

    # Append offspring to population
    pop += offspring_from_crossover
    pop += offspring_from_mutation

    # Evaluate fitness of the population
    fitness_values = evaluation(reduced_G, pop, speed_dict, cost_coefficients, profit_margins)

    # Selection process to maintain diversity
    pop = selection(pop, fitness_values, pop_size)

# Pareto front visualization
fitness_values = evaluation(reduced_G, pop, speed_dict, cost_coefficients, profit_margins)
index = np.arange(pop.shape[0]).astype(int)
pareto_front_index = pareto_front_finding(fitness_values, index)
pop = pop[pareto_front_index]
fitness_values = fitness_values[pareto_front_index]

index = np.arange(pop.shape[0]).astype(int)
pareto_front_index = pareto_front_finding(fitness_values, index)
pop = pop[pareto_front_index, :]
print("_________________")
print("Optimal solutions:")
print("       x1               x2                 x3")
print(pop)  # show optimal solutions
fitness_values = fitness_values[pareto_front_index]
print("______________")
print("Fitness values:")
print("  objective 1    objective 2")
print(fitness_values)
plt.scatter(fitness_values[:, 0], fitness_values[:, 1], label='Pareto optimal front')
plt.legend(loc='best')
plt.xlabel('Objective function F1')
plt.ylabel('Objective function F2')
plt.grid()
plt.show()