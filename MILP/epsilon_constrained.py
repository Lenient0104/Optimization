import random
import pulp
import time
import numpy as np  # 1.21.0
import random
# import matplotlib.pyplot as plt  # 3.3.2
from tqdm import tqdm
from energy_consumption_model import e_scooter
from energy_consumption_model import e_bike
from energy_consumption_model import e_car
import xml.etree.ElementTree as ET
import networkx as nx
import re
import json
import csv
import gurobipy as gp
from gurobipy import GRB
import time


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
                # num_preferred = random.randint(1, len(self.station_types) - 1)
                # preferred_types = [st for st in self.station_types if st != 'walk']
                # 为每个站点生成唯一的 station_types 列表
                # num_preferred = random.randint(1, len(self.station_types))  # 随机生成 1 到全部 station_types
                # preferred_types = random.sample(self.station_types, num_preferred)
                preferred_types = ['es', 'ec', 'walk']
                preferred_station[i] = preferred_types  # random.sample(preferred_types, num_preferred)
                if 'walk' not in preferred_station[i]:
                    preferred_station[i].append('walk')
                self.node_station_pair[i] = preferred_station[i]
            else:
                preferred_station[i] = ['']

        return preferred_station, preferred_nodes


class OptimizationProblem:
    def __init__(self, G, node_stations, preferred_station, M, speed_dict, user_preference, source, destination,
                 congestion=1):
        self.G = G
        self.node_stations = node_stations
        self.preferred_station = preferred_station
        self.M = M
        self.source = source
        self.destination = destination
        self.speed_dict = speed_dict
        self.user_preference = user_preference  # user_preference = ['eb', 'ec', 'es']
        self.congestion = congestion
        self.model = None
        self.total_time = None
        self.total_bike_time = None
        self.paths = {}
        self.energy_constraints = {}
        self.station_changes = {}
        self.costs = {}
        self.fees = {}
        self.energy_vars = {}
        self.station_change_costs = {}
        self.walk_distances = {}
        self.safety_scores = {}

    def setup_model(self):
        self.model = gp.Model("Minimize_Traversal_Cost")

    def setup_decision_variables(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Call setup_model() first.")

        for i, j in self.G.edges():
            stations = set(self.node_stations[i]).intersection(self.node_stations[j])
            if i != self.source and j != self.destination:
                stations = stations - {'walk'}
            for s in stations:
                var_name = f"path_{i}_{j}_{s}"
                self.paths[i, j, s] = self.model.addVar(vtype=GRB.BINARY, name=var_name)

        for i in self.G.nodes:
            for s1 in self.preferred_station[i]:
                for s2 in self.preferred_station[i]:
                    if s1 != s2:
                        var_name = f"station_change_{i}_{s1}_{s2}"
                        self.station_changes[i, s1, s2] = self.model.addVar(vtype=GRB.BINARY, name=var_name)

        self.total_ebike_time = self.model.addVar(vtype=GRB.CONTINUOUS, name="total_ebike_time")

        initial_energy = {  # in wh
            'eb': 50,
            'es': 50,
            'ec': 3000,
            'walk': 0
        }

        for i in self.G.nodes:
            self.energy_vars[i] = {}
            for s in self.node_stations[i]:
                var_name = f"energy_{i}_{s}"
                self.energy_vars[i][s] = self.model.addVar(vtype=GRB.CONTINUOUS, name=var_name)
                self.model.addConstr(self.energy_vars[i][s] == initial_energy[s], name=f"InitialEnergy_{i}_{s}")

    def setup_costs(self):
        for i, j in self.G.edges():
            edge_weight = self.G[i][j]['weight']
            edge_id = i
            speeds = self.speed_dict[edge_id]

            for s in set(self.node_stations[i]).intersection(self.node_stations[j]):
                if s != 'walk':
                    if s not in self.user_preference:
                        self.costs[i, j, s] = edge_weight * 1e7
                    else:
                        speed = speeds.get('bike_speed' if s in ['eb', 'es'] else 'car_speed', 0)
                        self.costs[i, j, s] = edge_weight / speed if speed != 0 else 1e7
                else:
                    pedestrian_speed = speeds.get('pedestrian_speed', 0)
                    self.costs[i, j, s] = edge_weight / pedestrian_speed if pedestrian_speed != 0 else 1e7

        for i in self.G.nodes:
            if len(self.preferred_station[i]) > 1:
                for s1 in self.preferred_station[i]:
                    for s2 in self.preferred_station[i]:
                        if s1 != s2:
                            self.station_change_costs[i, s1, s2] = 0.1
                        else:
                            self.station_change_costs[i, s1, s2] = self.M


    def set_up_walking_distance(self):
        for i, j in self.G.edges():
            edge_weight = self.G[i][j]['weight']
            for s in set(self.node_stations[i]).intersection(self.node_stations[j]):
                if s == 'walk':
                    self.walk_distances[i, j, s] = edge_weight
                elif s == 'eb' or s == 'es':
                    self.walk_distances[i, j, s] = (edge_weight ** 2) * 0.01
                else:
                    self.walk_distances[i, j, s] = (edge_weight ** 2) * 0.001

    def set_up_fees(self):
        for i, j in self.G.edges():
            for s in set(self.node_stations[i]).intersection(self.node_stations[j]):
                if s == 'ec':
                    energy_consumed_kWh = self.energy_constraints[i, j, s]
                    self.fees[i, j, s] = self.calculate_ecar_fees(energy_consumed_kWh, charging_type='standard')
                elif s == 'es':
                    ride_time_seconds = self.costs[i, j, s]
                    self.fees[i, j, s] = self.calculate_escooter_fees(ride_time_seconds / 3600)

    def calculate_ebike_total_time(self):
        total_ebike_time_expr = gp.quicksum(
            self.paths[i, j, s] * (self.costs[i, j, s] / 3600)
            for i, j, s in self.paths if s == 'eb'
        )
        return total_ebike_time_expr

    def calculate_flexible_pricing(self, pride, gamma=0.1, P0=0):
        total_ebike_time = self.total_ebike_time / 3600
        if total_ebike_time <= 0.5:
            base_price = 0
        elif total_ebike_time <= 1:
            base_price = 0.50
        elif total_ebike_time <= 2:
            base_price = 1.50
        elif total_ebike_time <= 3:
            base_price = 3.50
        elif total_ebike_time <= 4:
            base_price = 6.50
        else:
            extra_time = total_ebike_time - 4
            base_price = 6.50 + (extra_time // 0.5) * 2

        flexible_price = base_price + gamma * (pride - P0)

        return max(flexible_price, 0)

    def calculate_ecar_fees(self, energy_consumed_kWh, charging_type='standard'):
        price_per_kWh = {
            'standard': 0.52,  # €0.52/kWh
            'fast': 0.57,  # €0.57/kWh
            'high_power': 0.59  # €0.59/kWh
        }
        total_fees = energy_consumed_kWh * price_per_kWh[charging_type]
        return max(total_fees, 5)

    def calculate_escooter_fees(self, ride_time_seconds):
        unlock_fee = 0.15  # unlock fees
        ride_fee_per_min = 0.18  # (€0.18/min)
        ride_time_minutes = ride_time_seconds / 60
        total_fees = unlock_fee + ride_fee_per_min * ride_time_minutes

        return total_fees

    def set_up_risk(self):
        risky_level = {
            'es': 4,
            'eb': 3,
            'ec': 2,
            'walk': 1
        }
        for i, j in self.G.edges():
            edge_weight = self.G[i][j]['weight']
            # edge_id = i
            # speeds = self.speed_dict[edge_id]
            for s in set(self.node_stations[i]).intersection(self.node_stations[j]):
                self.safety_scores[i, j, s] = risky_level[s] * (edge_weight **2) * 0.000001

    def setup_energy_constraints(self, m, pal):
        for i, j in self.G.edges():
            edge_id = i
            speeds = self.speed_dict[edge_id]
            for s in set(self.node_stations[i]).intersection(self.node_stations[j]):
                if s != 'walk':
                    if s == 'es':
                        escooter_calculator = e_scooter.Escooter_PowerConsumptionCalculator()
                        self.energy_constraints[i, j, s] = escooter_calculator.calculate(
                            speeds['bike_speed'], m
                        )
                    elif s == 'eb':
                        ebike_calculator = e_bike.Ebike_PowerConsumptionCalculator()
                        self.energy_constraints[i, j, s] = ebike_calculator.calculate(
                            speeds['bike_speed'], m, pal)
                    else:
                        ecar_calculator = e_car.ECar_EnergyConsumptionModel(4)
                        self.energy_constraints[i, j, s] = ecar_calculator.calculate_energy_loss(
                            speeds['car_speed'])
                        if self.energy_constraints[i, j, s] == 0:
                            self.energy_constraints[i, j, s] = 60
                else:
                    self.energy_constraints[i, j, s] = 0

    def setup_problem(self, start_node, start_station, end_node, end_station, max_station_changes, reltol, pride, gamma=0.1, P0=0):
        # 添加约束，将 eBike 总路径时间与路径选择变量关联
        total_ebike_time_expr = self.calculate_ebike_total_time()

        self.model.addConstr(self.total_ebike_time == total_ebike_time_expr, name="ebike_time_constraint")
        # 引入辅助变量 z1, z2, z3, z4, z5, z6 用来表示时间区间
        z1 = self.model.addVar(vtype=GRB.BINARY, name="z1")  # 表示时间 <= 0.5
        z2 = self.model.addVar(vtype=GRB.BINARY, name="z2")  # 表示 0.5 < 时间 <= 1
        z3 = self.model.addVar(vtype=GRB.BINARY, name="z3")  # 表示 1 < 时间 <= 2
        z4 = self.model.addVar(vtype=GRB.BINARY, name="z4")  # 表示 2 < 时间 <= 3
        z5 = self.model.addVar(vtype=GRB.BINARY, name="z5")  # 表示 3 < 时间 <= 4
        z6 = self.model.addVar(vtype=GRB.BINARY, name="z6")  # 表示 时间 > 4

        # 约束只能落在一个区间
        self.model.addConstr(z1 + z2 + z3 + z4 + z5 + z6 == 1, name="time_zone_constraint")

        # 添加每个区间的时间约束
        self.model.addConstr(self.total_ebike_time <= 0.5 + 10 * (1 - z1), name="time_zone_1")  # 对应 z1
        self.model.addConstr(self.total_ebike_time <= 1 + 10 * (1 - z2), name="time_zone_2")  # 对应 z2
        self.model.addConstr(self.total_ebike_time <= 2 + 10 * (1 - z3), name="time_zone_3")  # 对应 z3
        self.model.addConstr(self.total_ebike_time <= 3 + 10 * (1 - z4), name="time_zone_4")  # 对应 z4
        self.model.addConstr(self.total_ebike_time <= 4 + 10 * (1 - z5), name="time_zone_5")  # 对应 z5
        self.model.addConstr(self.total_ebike_time >= 4 - 10 * (1 - z6), name="time_zone_6")  # 对应 z6

        # 新的辅助变量，用来表示 (self.total_ebike_time - 4) 的值
        extra_time = self.model.addVar(vtype=GRB.CONTINUOUS, name="extra_time")

        # 当时间超过4小时时，计算超过4小时的部分（每30分钟收费€2）
        self.model.addConstr(extra_time == (self.total_ebike_time - 4) * z6, name="extra_time_constraint")
        self.model.addConstr(extra_time >= 0, name="non_negative_extra_time")

        # 定义整数变量 extra_intervals 表示每 0.5 小时的增量
        extra_intervals = self.model.addVar(vtype=GRB.INTEGER, name="extra_intervals")

        # 添加线性约束，确保 extra_intervals 是 extra_time 的每 0.5 小时的整数倍
        self.model.addConstr(extra_intervals <= extra_time / 0.5, name="extra_intervals_upper_bound")
        self.model.addConstr(extra_intervals >= (extra_time - 0.5) / 0.5, name="extra_intervals_lower_bound")

        # 计算每 0.5 小时收费 €2
        extra_time_fees = 2 * extra_intervals

        # 定义阶梯费用
        # 注意：对于 z6 区间，我们只累加额外的 extra_time_fees，而不是再加上 6.5
        ebike_fees = 3.5 + (0.50 * z2) + (1.50 * z3) + (3.50 * z4) + (6.50 * z5) + extra_time_fees

        obj_fees_min = gp.quicksum(
            self.paths[i, j, s] * self.fees[i, j, s] for i, j, s in self.paths if s in ['ec', 'es']
        ) + ebike_fees  # 添加 eBike 的动态费用

        obj_time_min = gp.quicksum(self.paths[i, j, s] * self.costs[i, j, s] for i, j, s in self.paths) + \
                       gp.quicksum(self.station_changes[i, s1, s2] * self.station_change_costs[i, s1, s2] for i, s1, s2 in
                           self.station_changes)


        obj_safety_scores_min = gp.quicksum(self.paths[i, j, s] * self.safety_scores[i, j, s] for i, j, s in self.paths)

        self.model.ModelSense = gp.GRB.MINIMIZE

        # self.model.setObjective(obj_safety_scores_min, gp.GRB.MINIMIZE)
        self.model.setObjectiveN(obj_time_min, index=1, priority=3, reltol=reltol, name="Time")
        self.model.setObjectiveN(obj_fees_min, index=2, priority=2, name="Fees")
        self.model.setObjectiveN(obj_safety_scores_min, index=3, priority=1, name="risky")


        for i in self.G.nodes:
            for s in self.node_stations[i]:
                incoming_flow = gp.quicksum(
                    self.paths[j, i, s] for j in self.G.predecessors(i) if (j, i, s) in self.paths)
                outgoing_flow = gp.quicksum(
                    self.paths[i, j, s] for j in self.G.successors(i) if (i, j, s) in self.paths)

                incoming_station_changes = gp.quicksum(self.station_changes[i, s2, s] for s2 in self.node_stations[i] if
                                                       (i, s2, s) in self.station_changes)
                outgoing_station_changes = gp.quicksum(self.station_changes[i, s, s2] for s2 in self.node_stations[i] if
                                                       (i, s, s2) in self.station_changes)

                incoming_flow += incoming_station_changes
                outgoing_flow += outgoing_station_changes

                if i == start_node and s == start_station:
                    self.model.addConstr(outgoing_flow == 1, name=f"start_outflow_{i}_{s}")
                    self.model.addConstr(incoming_flow == 0, name=f"start_inflow_{i}_{s}")
                elif i == end_node and s == end_station:
                    self.model.addConstr(incoming_flow == 1, name=f"end_inflow_{i}_{s}")
                    self.model.addConstr(outgoing_flow == 0, name=f"end_outflow_{i}_{s}")
                else:
                    self.model.addConstr(incoming_flow == outgoing_flow, name=f"flow_balance_{i}_{s}")

        self.model.addConstr(gp.quicksum(self.station_changes.values()) <= max_station_changes,
                             name="max_station_changes")

        # energy reset
        initial_energy = {  # in wh
            'eb': 50,
            'es': 50,
            'ec': 3000,
            'walk': 0
        }
        for i in self.G.nodes:
            for s1 in self.preferred_station[i]:
                for s2 in self.preferred_station[i]:
                    if s1 != s2:
                        self.model.addConstr(
                            self.energy_vars[i][s2] >= self.station_changes[i, s1, s2] * initial_energy[s2],
                            name=f"EnergyReset_{i}_{s1}_{s2}"
                        )

        for i, j in self.G.edges():
            stations = set(self.node_stations[i]).intersection(self.node_stations[j])
            if i != self.source and j != self.destination:
                stations = stations - {'walk'}
            for s in stations:
                energy_consumption = self.energy_constraints[i, j, s]

                self.model.addConstr(
                    self.paths[i, j, s] * energy_consumption <= self.energy_vars[i][s],
                    name=f"PathEnergyFeasibility_{i}_{j}_{s}"
                )
                # print(self.energy_vars[i][s])
                self.model.addConstr(
                    self.energy_vars[j][s] >= self.energy_vars[i][s] - energy_consumption * self.paths[i, j, s],
                    name=f"EnergyConsumption_{i}_{j}_{s}"
                )
                self.model.update()
                # print(self.energy_vars[i][s])

    def solve(self):
        start_time = time.time()

        self.model.optimize()
        end_time = time.time()

        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        elif self.model.status == GRB.UNBOUNDED:
            print("Model is unbounded.")
        else:
            print(f"Optimization ended with status {self.model.status}")

        return self.model, end_time - start_time


class PathFinder:
    def __init__(self, G, paths, station_changes, costs, station_change_costs, energy_constraints):
        self.G = G
        self.paths = paths  # Gurobi decision variables for paths
        self.station_changes = station_changes  # Gurobi decision variables for station changes
        self.costs = costs  # Costs associated with paths
        self.station_change_costs = station_change_costs  # Costs associated with station changes
        self.energy_constraints = energy_constraints

    def calculate_fees(self, path):
        fees = 0
        cost_coefficients = {
            'ec': 0.9,  # 电动汽车每公里成本
            'eb': 0.2,  # 电动自行车每公里成本
            'es': 0.1,  # 电动滑板车每公里成本
            'walk': 0  # 步行每公里成本
        }

        # 利润率
        profit_margins = {
            'ec': 1.8,
            'eb': 0.10,
            'es': 0.12,
            'walk': 0
        }

        for sequence in path:
            if len(sequence) == 5:
                i = sequence[0]
                j = sequence[1]
                s = sequence[2]
                dis = self.G[i][j]['weight'] / 1000
                base_cost = cost_coefficients[s] * dis
                fees = fees + base_cost + profit_margins[s] * dis

        return fees

    def calculate_time(self, path):
        time_cost = 0
        for sequence in path:
            t = sequence[3]
            time_cost = time_cost + t
        return time_cost

    def calculate_safety_level(self, path):
        safety = 0
        safety_level = {
            'es': 4,
            'eb': 3,
            'ec': 2,
            'walk': 1
        }
        for sequence in path:
            if len(sequence) == 5:
                i = sequence[0]
                j = sequence[1]
                s = sequence[2]
                dis = self.G[i][j]['weight']
                safety = safety + safety_level[s] * (dis ** 2) * 0.000001
        return safety

    def calculate_walking_distance(self, path):
        walking_distance = 0
        for sequence in path:
            if len(sequence) == 5:
                i = sequence[0]
                j = sequence[1]
                s = sequence[2]
                dis = self.G[i][j]['weight']
                if s == 'walk':
                    walking_distance = walking_distance + dis
                elif s == 'eb' or s == 'es':
                    walking_distance = walking_distance + dis * 0.01
                else:
                    walking_distance = walking_distance + dis * 0.001

        return walking_distance

    def generate_path_sequence(self, start_node, start_station, end_node, end_station):
        current_node, current_mode = start_node, start_station
        path_sequence = []
        energy_consumption_sequence = []  # List to store energy consumption
        station_change_count = 0
        destination_reached = False

        while not destination_reached:
            next_step_found = False

            # Look for the next path step
            for (i, j, s) in self.paths:
                if i == current_node and s == current_mode and self.paths[i, j, s].X == 1:
                    path_cost = self.costs[i, j, s]
                    energy_consumption = self.energy_constraints[i, j, s]
                    path_sequence.append((i, j, s, path_cost, energy_consumption))
                    energy_consumption_sequence.append((i, j, s, energy_consumption))
                    current_node = j
                    next_step_found = True
                    break

            # Look for the next station change
            for (i, s1, s2) in self.station_changes:
                if i == current_node and s1 == current_mode and self.station_changes[i, s1, s2].X == 1:
                    mode_change_cost = self.station_change_costs[i, s1, s2]
                    path_sequence.append((i, s1, s2, mode_change_cost))
                    current_mode = s2
                    station_change_count += 1
                    next_step_found = True

            # Check if destination is reached
            if current_node == end_node and current_mode == end_station:
                destination_reached = True
            elif not next_step_found:
                print("Destination not reached. Path may be incomplete.")
                break

        return path_sequence, station_change_count, self.calculate_fees(path_sequence), \
               self.calculate_time(path_sequence), self.calculate_safety_level(path_sequence), self.calculate_walking_distance(path_sequence)


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


######################O*****************Original END *********************
##########################################################################


################PipeLine to Execute the OD pairs from CSV ###################
pareto_values = "pareto_values1015.csv"
with open(pareto_values, 'w') as pafile:
    for rel in [0.08]:
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

                            path_finder = PathFinder(reduced_G, optimization_problem.paths,
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