from energy_consumption_model import e_scooter
from energy_consumption_model import e_bike
from energy_consumption_model import e_car
import gurobipy as gp
from gurobipy import GRB
import time


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
        self.energy = {}
        self.station_changes = {}
        self.costs = {}
        self.fees = {}
        self.energy_vars = {}
        self.station_change_costs = {}
        self.walk_distances = {}
        self.safety_scores = {}
        self.vehicle_info = {}
        self.current_vehicle = None  # 用来记录当前使用的车辆


    def setup_model(self):
        self.model = gp.Model("Minimize_Traversal_Cost")
        self.total_ebike_time = self.model.addVar(vtype=GRB.CONTINUOUS, name="total_ebike_time")

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
            for s1 in self.preferred_station[i]['types']:
                for s2 in self.preferred_station[i]['types']:
                    if s1 != s2:
                        var_name = f"station_change_{i}_{s1}_{s2}"
                        self.station_changes[i, s1, s2] = self.model.addVar(vtype=GRB.BINARY, name=var_name)

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
            if len(self.preferred_station[i]['types']) > 1:
                for s1 in self.preferred_station[i]['types']:
                    for s2 in self.preferred_station[i]['types']:
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
                    energy_consumed_kWh = self.energy[i, j, s]
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

    def setup_energy(self, m, pal):
        for i, j in self.G.edges():
            edge_id = i
            speeds = self.speed_dict[edge_id]
            for s in set(self.node_stations[i]).intersection(self.node_stations[j]):
                if s != 'walk':
                    if s == 'es':
                        escooter_calculator = e_scooter.Escooter_PowerConsumptionCalculator()
                        self.energy[i, j, s] = escooter_calculator.calculate(
                            speeds['bike_speed'], m
                        )
                    elif s == 'eb':
                        ebike_calculator = e_bike.Ebike_PowerConsumptionCalculator()
                        self.energy[i, j, s] = ebike_calculator.calculate(
                            speeds['bike_speed'], m, pal)
                    else:
                        ecar_calculator = e_car.ECar_EnergyConsumptionModel(4)
                        self.energy[i, j, s] = ecar_calculator.calculate_energy_loss(
                            speeds['car_speed'])
                        if self.energy[i, j, s] == 0:
                            self.energy[i, j, s] = 60
                else:
                    self.energy[i, j, s] = 0

    def setup_max_energy_constraints(self):
        # 初始化能量变量并设置为站点中最大能量的车辆的能量
        for i in self.G.nodes:
            self.energy_vars[i] = {}  # 初始化每个节点的能量字典
            for s in self.node_stations[i]:
                var_name = f"energy_{i}_{s}"
                self.energy_vars[i][s] = self.model.addVar(vtype=GRB.CONTINUOUS, name=var_name)

                # 找到该站点中最大能量的车辆
                max_energy = 0
                for vehicle in self.preferred_station[i]['vehicles']:
                    if vehicle['type'] == s:
                        # 更新 max_energy 为车辆中的最大电量
                        if vehicle['battery'] > max_energy:
                            max_energy = vehicle['battery']

                # 将 self.energy_vars[i][s] 初始化为站点中最大能量车辆的电量
                self.model.addConstr(self.energy_vars[i][s] == max_energy, name=f"InitialEnergy_{i}_{s}")

    def setup_energy_constraints(self):
        initial_energy = {  # 初始电量（以 wh 为单位）
            'eb': 500,
            'es': 500,
            'ec': 30000,
            'walk': 0
        }

        # 遍历每条路径段，处理能量流动和使用约束
        for i, j in self.G.edges():
            stations = set(self.node_stations[i]).intersection(self.node_stations[j])
            if i != self.source and j != self.destination:
                stations = stations - {'walk'}

                for s in stations:
                    energy_consumption = self.energy.get((i, j, s), 0)

                    self.model.addConstr(
                        self.paths[i, j, s] * energy_consumption <= self.energy_vars[i][s],
                        name=f"PathEnergyFeasibility_{i}_{j}_{s}"
                    )

                    # 判断换乘情况：如果 s1 != s2，需要换乘
                    for s1 in self.preferred_station[i]['types']:
                        for s2 in self.preferred_station[i]['types']:
                            if s1 != s2:
                                # 获取站点中车辆的能量信息，找到能量最小且满足条件的车辆
                                min_energy_vehicles = [vehicle for vehicle in self.preferred_station[i]['vehicles'] if
                                                       vehicle['type'] == s and vehicle['battery'] >= energy_consumption]

                                # 只有至少一辆车可以满足能量消耗时，才允许换乘并重置能量
                                if min_energy_vehicles:
                                    max_energy = max([vehicle['battery'] for vehicle in min_energy_vehicles])
                                    # 确保站点有至少一辆符合条件的车辆
                                    self.model.addConstr(
                                        len(min_energy_vehicles) >= 1,
                                        name=f"ValidVehicleAvailable_{i}_{j}_{s}"
                                    )
                                    # 能量消耗必须小于最小车辆的能量
                                    self.model.addConstr(
                                        self.station_changes[i, s1, s2] * energy_consumption <= max_energy,
                                        name=f"PathEnergyFeasibilityWithSwitch_{i}_{j}_{s}"
                                    )
                                    # 能量重置：新交通工具的能量重置为初始值
                                    self.model.addConstr(
                                        self.energy_vars[i][s2] >= self.station_changes[i, s1, s2] * initial_energy[s2],
                                        name=f"EnergyReset_{i}_{s1}_{s2}"
                                    )

                    # 能量流动约束：确保从站点 i 到站点 j 的能量流动是正确的
                    # 这里包含了不换乘的情况
                    self.model.addConstr(
                        self.energy_vars[j][s] >= self.energy_vars[i][s] - energy_consumption * self.paths[i, j, s],
                        name=f"EnergyFlow_{i}_{j}_{s}"
                    )

    def setup_problem(self, start_node, start_station, end_node, end_station, max_station_changes, reltol, pride, gamma=0.01, P0=0):
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
        self.model.addConstr(self.total_ebike_time <= 0.05 + 10 * (1 - z1), name="time_zone_1")  # 对应 z1
        self.model.addConstr(self.total_ebike_time <= 0.1 + 10 * (1 - z2), name="time_zone_2")  # 对应 z2
        self.model.addConstr(self.total_ebike_time <= 0.2 + 10 * (1 - z3), name="time_zone_3")  # 对应 z3
        self.model.addConstr(self.total_ebike_time <= 0.3 + 10 * (1 - z4), name="time_zone_4")  # 对应 z4
        self.model.addConstr(self.total_ebike_time <= 0.4 + 10 * (1 - z5), name="time_zone_5")  # 对应 z5
        self.model.addConstr(self.total_ebike_time >= 0.4 - 10 * (1 - z6), name="time_zone_6")  # 对应 z6

        # 新的辅助变量，用来表示 (self.total_ebike_time - 0.4) 的值
        extra_time = self.model.addVar(vtype=GRB.CONTINUOUS, name="extra_time")

        # 当时间超过4小时时，计算超过0.4小时的部分（每30分钟收费€0.2）
        self.model.addConstr(extra_time == (self.total_ebike_time - 0.4) * z6, name="extra_time_constraint")
        self.model.addConstr(extra_time >= 0, name="non_negative_extra_time")

        # 定义整数变量 extra_intervals 表示每 0.5 小时的增量
        extra_intervals = self.model.addVar(vtype=GRB.INTEGER, name="extra_intervals")

        # 添加线性约束，确保 extra_intervals 是 extra_time 的每 0.5 小时的整数倍
        self.model.addConstr(extra_intervals <= extra_time / 0.5, name="extra_intervals_upper_bound")
        self.model.addConstr(extra_intervals >= (extra_time - 0.5) / 0.5, name="extra_intervals_lower_bound")

        # 计算每 0.5 小时收费 €0.2
        extra_time_fees = 0.2 * extra_intervals

        ebike_fees_total = gp.quicksum(
            self.paths[i, j, 'eb'] for i, j in self.G.edges() if
            'eb' in self.node_stations[i] and 'eb' in self.node_stations[j]
        )

        # 定义阶梯费用
        # 注意：对于 z6 区间，我们只累加额外的 extra_time_fees，而不是再加上 6.5
        # 修改 ebike_fees，使得 3.5 只在选择了 eBike 且路径存在时计入
        ebike_fees = ebike_fees_total * 0.01 + (0.50 * z2) + (1.50 * z3) + (3.50 * z4) + (6.50 * z5) + extra_time_fees + gamma * (pride - P0)

        obj_fees_min = gp.quicksum(
            self.paths[i, j, s] * self.fees[i, j, s] for i, j, s in self.paths if s in ['ec', 'es']
        ) + ebike_fees

        obj_time_min = gp.quicksum(self.paths[i, j, s] * self.costs[i, j, s] for i, j, s in self.paths) + \
                       gp.quicksum(self.station_changes[i, s1, s2] * self.station_change_costs[i, s1, s2] for i, s1, s2 in
                           self.station_changes)


        obj_safety_scores_min = gp.quicksum(self.paths[i, j, s] * self.safety_scores[i, j, s] for i, j, s in self.paths)

        self.model.ModelSense = gp.GRB.MINIMIZE


        self.model.setObjective(obj_time_min, gp.GRB.MINIMIZE)
        # self.model.setObjectiveN(obj_time_min, index=1, priority=3, reltol=reltol, name="Time")
        # self.model.setObjectiveN(obj_fees_min, index=2, priority=2, name="Fees")
        # self.model.setObjectiveN(obj_safety_scores_min, index=3, priority=1, name="risky")


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
