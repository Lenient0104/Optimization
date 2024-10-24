import threading

import pulp
import json
from user import User
from graphhandler import GraphHandler
from preference_generator import PreferenceGenerator
from shortestpath_computer import ShortestPathComputer
from reduced_graph import ReducedGraphCreator
from optimization_problem import OptimizationProblem
from path_generator import PathGenerator
import csv
from gurobipy import GRB
import time


class RunMilp:
    def __init__(self):
        file_path = "DCC.net.xml"
        speed_file_path = 'query_results-0.json'

        # Create graph from XML file
        graph_handler = GraphHandler(file_path)
        self.original_G = graph_handler.get_graph()
        self.num_nodes = len(self.original_G.nodes)
        self.no_pref_nodes = 10
        self.M = 1e6
        # route_finder = RouteFinder(original_G)
        # Generate preferred station types for each node (execute only once)
        station_types = ['eb', 'es', 'ec', 'walk']
        preference_generator = PreferenceGenerator(self.original_G, station_types)
        self.preferred_station, self.preferred_nodes = preference_generator.generate_node_preferences()

        # Compute shortest routes pairs (execute only once)
        self.shortest_path_computer = ShortestPathComputer(self.original_G)
        # all_shortest_routes_pairs = shortest_path_computer.compute_shortest_paths_pairs(preferred_nodes)
        # Load speed data from JSON
        with open(speed_file_path, 'r') as f:
            self.speed_data = json.load(f)

        # Create a dictionary for speed data
        self.speed_dict = {entry['edge_id']: {'pedestrian_speed': float(entry['pedestrian_speed']),
                                              'bike_speed': float(entry['bike_speed']),
                                              'car_speed': float(entry['car_speed'])}
                           for entry in self.speed_data}
        self.resource_lock = threading.Lock()

    def user_task(self, user: User):
        self.resource_lock.acquire()
        try:
            path = self.optimize_with_gurobi(user)[0]
            self.update_preferred_station(path)
        finally:
            self.resource_lock.release()

    def optimize_with_gurobi(self, user):
        output_csv_file = 'multi-users.csv'
        # Prepare the output CSV file
        with open(output_csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Start Node', 'End Node', 'Total Cost', 'Execution Time', 'Optimal Path Sequence',
                             'Final Route Mapping to original'])

            initial_time = time.time()
            start_node, end_node = user.source, user.destination
            all_shortest_routes_pairs = self.shortest_path_computer.compute_shortest_paths_pairs(self.preferred_nodes)

            # Compute shortest routes start
            shortest_routes_start = self.shortest_path_computer.compute_shortest_paths_start(start_node,
                                                                                             self.preferred_nodes)

            # Compute shortest routes dest
            shortest_routes_dest = self.shortest_path_computer.compute_shortest_paths_dest(end_node,
                                                                                           self.preferred_nodes)

            # compute shortest route between start and end
            # shortest_route_start_end = shortest_path_computer.compute_shortest_path_start_end(start_node, end_node)
            # Create a new reduced graph
            reduced_graph_creator = ReducedGraphCreator(self.original_G, start_node, end_node, self.preferred_nodes,
                                                        shortest_routes_start, shortest_routes_dest,
                                                        all_shortest_routes_pairs)
            # reduced_graph_creator = ReducedGraphCreator(original_G, start_node, end_node, preferred_nodes,
            #                                             shortest_routes_start, shortest_routes_dest,
            #                                             all_shortest_routes_pairs, shortest_route_start_end)
            reduced_G = reduced_graph_creator.create_new_graph()

            # Set up and solve the optimization problem
            optimization_problem = OptimizationProblem(reduced_G, user.node_stations, self.preferred_station, self.M,
                                                       self.speed_dict,
                                                       user.user_preference, start_node, end_node)
            rel = 0
            optimization_problem.setup_model()
            optimization_problem.setup_decision_variables()
            optimization_problem.setup_costs()
            optimization_problem.set_up_risk()
            optimization_problem.setup_energy(50, 1)
            optimization_problem.setup_max_energy_constraints()
            optimization_problem.setup_energy_constraints()
            optimization_problem.set_up_fees()
            optimization_problem.setup_problem(start_node, 'walk', end_node, 'walk', user.max_station_changes, rel, 20)
            path_sequence = None
            station_change_count = None
            fees = None
            total_time = None
            safety = None
            walking_distance = None

            try:
                # prob, execution_time = optimization_problem.solve()
                optimization_problem.solve()
                # optimization_problem.model.computeIIS()
                # optimization_problem.model.write("model.ilp")
                # optimization_problem.model.write("mymodel.lp")

                if optimization_problem.model.status == GRB.OPTIMAL:
                    total_cost = optimization_problem.model.getObjective().getValue()
                    # total_cost = pulp.value(prob.objective)
                    path_finder = PathGenerator(reduced_G, optimization_problem.paths,
                                                optimization_problem.station_changes,
                                                optimization_problem.costs,
                                                optimization_problem.station_change_costs,
                                                optimization_problem.energy)
                    path_sequence, station_change_count, fees, total_time, safety, walking_distance = path_finder.generate_path_sequence(
                        start_node, 'walk', end_node, 'walk')
                    # obj_values.append(path_sequence)

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
            return [path_sequence, station_change_count, fees, total_time, safety, walking_distance]

    # 更新 preferred_station 中的车辆电量
    def update_preferred_station(self, path):
        initial_energy = {  # in wh
            'eb': 50,
            'es': 50,
            'ec': 3000,
            'walk': 0
        }
        current_vehicle = None  #
        # 遍历路径中的每一段
        for step in path:
            if len(step) == 5:
                start_node, end_node, transport_type, time_cost, energy_consumption = step

                # 处理取车的逻辑：从 walk 切换到车辆时，取出该车辆
                if current_vehicle is None and transport_type in ['ec', 'es', 'eb']:
                    # 获取站点的车辆列表
                    vehicles = self.preferred_station[start_node]['vehicles']
                    for idx, vehicle in enumerate(vehicles):
                        if vehicle['type'] == transport_type and vehicle['battery'] > energy_consumption:
                            current_vehicle = vehicle  # 取出车辆
                            vehicle['battery'] -= energy_consumption  # 更新车辆电量
                            self.preferred_station[start_node]['vehicles'].pop(idx)
                            print(
                                f"Vehicle {vehicle['id']} taken from station {start_node}. Battery after usage: {vehicle['battery'] / initial_energy[transport_type]}%")
                            break

                # 处理换乘逻辑：从一种车辆切换到另一种车辆时，还车并取新车
                elif current_vehicle is not None and transport_type in ['ec', 'es', 'eb'] and current_vehicle[
                    'type'] != transport_type:
                    current_type = current_vehicle['type']
                    # 先还车
                    print(
                        f"Vehicle {current_vehicle['id']} returned to station {start_node}. Final battery: {current_vehicle['battery'] / initial_energy[current_type]}%")
                    self.preferred_station[start_node]['vehicles'].append(current_vehicle)  # 将车还回站点
                    # 再取新车
                    vehicles = self.preferred_station[start_node]['vehicles']
                    for idx, vehicle in enumerate(vehicles):
                        if vehicle['type'] == transport_type and vehicle['battery'] > energy_consumption:
                            current_vehicle = vehicle  # 取出新车
                            vehicle['battery'] -= energy_consumption  # 更新电量
                            # 从站点中移除该车辆
                            self.preferred_station[start_node]['vehicles'].pop(idx)
                            print(
                                f"New vehicle {vehicle['id']} taken from station {start_node}. Battery after usage: {vehicle['battery']/initial_energy[transport_type]}%")
                            break

                # 处理还车逻辑：当从车辆切换回步行时，将车辆还回站点
                elif current_vehicle is not None and transport_type == 'walk':
                    current_type = current_vehicle['type']
                    print(
                        f"Vehicle {current_vehicle['id']} returned to station {start_node}. Final battery: {current_vehicle['battery']/initial_energy[current_type]}%")
                    self.preferred_station[start_node]['vehicles'].append(current_vehicle)  # 将车还回站点
                    current_vehicle = None  # 清空当前车辆

                # 没有换乘，但使用同一辆车时只减少能量
                elif current_vehicle is not None and current_vehicle['type'] == transport_type:
                    current_vehicle['battery'] -= energy_consumption  # 更新电量
                    current_type = current_vehicle['type']
                    print(
                        f"Vehicle {current_vehicle['id']} at station {start_node} used. Battery after usage: {current_vehicle['battery']/initial_energy[current_type]}%")

            elif len(step) == 4:  # 不包含电量消耗信息的路径段
                start_node, from_type, to_type, transfer_cost = step
                print(f"Transition from {from_type} to {to_type} at station {start_node}.")

    def run_user(self):
        user_A = User(1, ['eb', 'es', 'ec'], "361450282", "-110407380#1", 3, self.original_G)
        user_B = User(2, ['eb', 'es', 'ec'], "361450282", "-110407380#1", 3, self.original_G)
        thread_A = threading.Thread(target=self.user_task, args=(user_A,))
        thread_B = threading.Thread(target=self.user_task, args=(user_B,))
        thread_A.start()
        thread_B.start()

        thread_A.join()
        thread_B.join()

        print("All tasks completed.")


if __name__ == "__main__":
    runner = RunMilp()
    runner.run_user()
