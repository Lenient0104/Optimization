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
