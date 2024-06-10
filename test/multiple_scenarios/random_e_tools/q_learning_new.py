import csv
import time
import networkx as nx
import matplotlib.pyplot as plt
import unittest
from algorithms.Q_learning import Q_learning_agent
from optimization_interface.user_info import User
from optimization_interface.optimization import Optimization


class TestDQN(unittest.TestCase):
    def setUp(self):
        self.net_xml_path = '../../../optimization_interface/DCC.net.xml'
        self.start_mode = 'walking'
        self.station_num = [20]
        self.energy_rate = [1]
        self.simulation_time = [20000]
        self.episodes = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        self.iteration = 1
        self.db_path = '../../../optimization_interface/test_new.db'
        self.user = User(60, True, 0, 20)
        self.done = False

    def test_run_dqn(self):
        with open('../../../optimization_interface/od_pairs/od_pairs_500_new.csv', 'r') as file:
            reader = csv.reader(file)
            od_pairs = [tuple(row) for row in reader]

        test_size = len(od_pairs)
        test_size = 30
        # test_od_pairs = ('3191574', '22770275#2')

        with open('results/Q-learning-parameters.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Experiment ID', 'Episode', 'Simulation Time', 'Station Number', 'Initial Energy', 'Travel Time Cost (seconds)',
                             'Execution Time (seconds)', 'Modes', 'Path', 'Find'])

            all_Q_Learning_exe_times = []
            all_Q_Learning_times = []

            # Perform experiments
            # for energy in self.energy_rate:
            #     episode_exe_times = []
            #     episode_times = []
            #     successful_tests = 0
            for episode in self.episodes:
                for test_index in range(test_size): # 500 od pairs
                    print(test_index)
                    source_edge, target_edge = od_pairs[0]
                    # source_edge, target_edge = test_od_pairs
                    optimizer = Optimization(self.net_xml_path, self.user, self.db_path, self.simulation_time[0], self.station_num[0], source_edge,
                                             target_edge)
                    graph = optimizer.new_graph
                    # self.visualize_graph(graph)
                    if graph is None:
                        writer.writerow([test_size + 1, episode, self.simulation_time[0], self.station_num[0], self.energy_rate[0], 0, 0, None, False])
                        continue
                    best_route, best_modes, total_time_cost, execution_time, find = Q_learning_agent.run_q_learning(
                        optimizer, source_edge, target_edge, episode, self.energy_rate[0])

                    print(best_route)
                    print(best_modes)
                    print(total_time_cost)
                    print(find)
                    # Write each test's result to the CSV file
                    experiment_id = f"{episode}-{test_index}"
                    writer.writerow([experiment_id, episode, self.simulation_time[0], self.station_num[0], self.energy_rate[0], total_time_cost, execution_time,best_modes, best_route, find])

                        # if find:
                        #     episode_exe_times.append(execution_time)
                        #     episode_times.append(total_time_cost)
                        #     successful_tests += 1

                    # all_Q_Learning_exe_times.append(episode_exe_times)
                    # all_Q_Learning_times.append(episode_times)
