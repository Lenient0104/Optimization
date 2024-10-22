import pulp
import json
from graphhandler import GraphHandler
from preference_generator import PreferenceGenerator
from shortestpath_computer import ShortestPathComputer
from reduced_graph import ReducedGraphCreator
from optimization_problem import OptimizationProblem
from path_generator import PathGenerator
import csv
from gurobipy import GRB
import time


pareto_values = "pareto_values1017new.csv"
with open(pareto_values, 'w') as pafile:
    for rel in [0.01, 0.05, 0.07, 0.08, 0.2, 0.3, 0.4, 0.5, 0.6]:
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
        max_station_changes = 5
        M = 1e6

        # route_finder = RouteFinder(original_G)


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

        # Prepare the output CSV file
        with open(output_csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Start Node', 'End Node', 'Total Cost', 'Execution Time', 'Optimal Path Sequence',
                             'Final Route Mapping to original'])

            # Read OD pairs from CSV file and execute the main loop
            with open(od_pairs_file, 'r') as odfile:
                reader = csv.reader(odfile)
                # next(reader)  # Skip header row if present
                for row in reader:

                    initial_time = time.time()

                    start_node, end_node = row  # Extract start_node and end_node from the current row

                    # Compute shortest route pairs for testing only - otherwise outside the loop
                    all_shortest_routes_pairs = shortest_path_computer.compute_shortest_paths_pairs(preferred_nodes)

                    # Compute shortest routes start
                    shortest_routes_start = shortest_path_computer.compute_shortest_paths_start(start_node, preferred_nodes)

                    # Compute shortest routes dest
                    shortest_routes_dest = shortest_path_computer.compute_shortest_paths_dest(end_node, preferred_nodes)

                    # compute shortest route between start and end
                    # shortest_route_start_end = shortest_path_computer.compute_shortest_path_start_end(start_node, end_node)
                    # Create a new reduced graph
                    reduced_graph_creator = ReducedGraphCreator(original_G, start_node, end_node, preferred_nodes,
                                                                shortest_routes_start, shortest_routes_dest,
                                                                all_shortest_routes_pairs)
                    # reduced_graph_creator = ReducedGraphCreator(original_G, start_node, end_node, preferred_nodes,
                    #                                             shortest_routes_start, shortest_routes_dest,
                    #                                             all_shortest_routes_pairs, shortest_route_start_end)
                    reduced_G = reduced_graph_creator.create_new_graph()

                    # Set up and solve the optimization problem
                    optimization_problem = OptimizationProblem(reduced_G, node_stations, preferred_station, M, speed_dict,
                                                               user_preference, start_node, end_node)
                    # rel = 0
                    optimization_problem.setup_model()
                    optimization_problem.setup_decision_variables()
                    optimization_problem.setup_costs()
                    # optimization_problem.set_up_walking_distance()
                    optimization_problem.set_up_risk()
                    optimization_problem.setup_energy_constraints(50, 1)
                    optimization_problem.set_up_fees()
                    optimization_problem.setup_problem(start_node, 'walk', end_node, 'walk', max_station_changes, rel, 20)

                    try:
                        # Solve the problem and measure execution time
                        prob, execution_time = optimization_problem.solve()
                        optimization_problem.model.write("mymodel.lp")

                        if optimization_problem.model.status == GRB.OPTIMAL:
                            num_of_objectives = optimization_problem.model.NumObj  # 获取目标函数的数量
                            obj_values = []
                            for i in range(num_of_objectives):
                                obj_value = optimization_problem.model.getObjective(i).getValue()
                                obj_values.append(obj_value)
                                print(f"Objective {i} value: {obj_value}")
                            obj_values.append(rel)

                            total_cost = optimization_problem.model.getObjective().getValue()

                            # total_cost = pulp.value(prob.objective)

                            path_finder = PathGenerator(reduced_G, optimization_problem.paths,
                                                        optimization_problem.station_changes,
                                                        optimization_problem.costs,
                                                        optimization_problem.station_change_costs,
                                                        optimization_problem.energy_constraints)
                            path_sequence, station_change_count, fees, total_time, safety, walking_distance = path_finder.generate_path_sequence(start_node, 'walk', end_node, 'walk')
                            obj_values.append(path_sequence)
                            writer_1 = csv.writer(pafile)
                            writer_1.writerow(obj_values)
                            # print("time:", total_time)
                            # print("fees:", fees)
                            # print("risky:", safety)
                            # print("walking distance", walking_distance)

                            end_time = time.time()
                            Total_time = end_time - initial_time

                            # Write results to CSV
                            writer.writerow([start_node, end_node, total_cost, Total_time, path_sequence])
                        else:
                            # Write results to CSV with 'inf' for total cost if no optimal solution is found
                            writer.writerow([start_node, end_node, 'inf', 'No optimal solution found'])
                    except pulp.PulpSolverError:
                        # Write results to CSV with 'inf' for total cost if solver fails
                        writer.writerow([start_node, end_node, 'inf', 'Solver failed'])