import csv
import unittest
from user_info import User
import matplotlib.pyplot as plt
from optimization import Optimization


class Analysis(unittest.TestCase):
    def setUp(self):
        self.net_xml_path = './medium/graph202311211622.net.xml'
        self.start_mode = 'walking'
        self.ant_num = [100, 300, 500, 700, 900]
        self.iteration = 1
        self.db_path = 'test_new.db'
        self.user = User(60, True, 0, 20)

    def test_run_optimizer(self):
        with open('od_pairs_50.csv', 'r') as file:
            reader = csv.reader(file)
            od_pairs = [tuple(row) for row in reader]

        test_size = len(od_pairs)
        all_aco_exe_time_costs = []
        all_aco_time_costs = []

        with open('results/test', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Experiment ID', 'Number of Ants', 'Travel Time Cost (seconds)', 'Execution Time (seconds)', 'Find'])

            for ant_num in self.ant_num:
                aco_time_costs = []
                aco_exe_costs = []
                for i in range(test_size):
                    print(ant_num, i)
                    source_edge, target_edge = od_pairs[i]
                    optimizer_interface = Optimization(self.net_xml_path, self.user, self.db_path, source_edge, target_edge)
                    graph = optimizer_interface.new_graph
                    if graph is None:
                        writer.writerow([i + 1, ant_num, 0, 0, False])
                        continue
                    path, time_cost, exe_time = optimizer_interface.run_aco_algorithm(source_edge, target_edge, ant_num)
                    print(time_cost, path)
                    all_aco_time_costs.append(time_cost)
                    all_aco_exe_time_costs.append(exe_time)
                    experiment_id = f"{ant_num}-{i + 1}"
                    writer.writerow([experiment_id, ant_num, time_cost, exe_time, 'True'])




if __name__ == '__main__':
    unittest.main()
