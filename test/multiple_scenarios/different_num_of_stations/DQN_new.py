import csv
import unittest
from algorithms.DQN import DQN
from optimization_interface.user_info import User
from optimization_interface.optimization import Optimization


class TestDQN(unittest.TestCase):
    def setUp(self):
        self.net_xml_path = '../../../optimization_interface/DCC.net.xml'
        self.start_mode = 'walking'
        self.station_num = [5, 10, 15, 20]
        self.simulation_time = [20000]
        self.episodes = [400]
        self.iteration = 1
        self.db_path = '../../../optimization_interface/test_new.db'
        self.user = User(60, True, 0, 20)
        self.done = False

    def test_run_dqn(self):
        with open('../../../optimization_interface/od_pairs/od_pairs_500_new.csv', 'r') as file:
            reader = csv.reader(file)
            od_pairs = [tuple(row) for row in reader]

        test_size = len(od_pairs)
        # test_size = 1

        with open('results/DQN-new-station.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Experiment ID', 'Episode', 'Simulation Time', 'Station Number', 'Travel Time Cost (seconds)',
                 'Execution Time (seconds)', 'Find'])

            all_DQN_exe_times = []
            all_DQN_times = []
            all_successful_tests = []

            # Perform experiments
            for station in self.station_num:
                episode_exe_times = []
                episode_times = []
                successful_tests = 0

                for test_index in range(test_size): # 500 od pairs
                    print(station, test_index)
                    source_edge, target_edge = od_pairs[test_index]
                    optimizer = Optimization(self.net_xml_path, self.user, self.db_path, self.simulation_time[0],
                                             station, source_edge,
                                             target_edge)
                    graph = optimizer.new_graph
                    if graph is None:
                        writer.writerow([test_size + 1, self.episodes[0], self.simulation_time[0], station, 0, 0, False])
                        continue
                    best_route, best_modes, total_time_cost, execution_time, find = DQN.run_dqn(optimizer, source_edge,
                                                                                                     target_edge, self.episodes[0])
                    print(best_route)
                    print(best_modes)
                    print(total_time_cost)
                    # Write each test's result to the CSV file
                    experiment_id = f"{self.episodes[0]}-{test_index}"
                    writer.writerow([experiment_id, self.episodes[0], self.simulation_time[0], station, total_time_cost, execution_time, find])

                    if find:
                        episode_exe_times.append(execution_time)
                        episode_times.append(total_time_cost)
                        successful_tests += 1

                all_DQN_exe_times.append(episode_exe_times)
                all_DQN_times.append(episode_times)




